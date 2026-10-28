"""Synthetic correctness fixtures; never reported as observed benchmark data."""
import json
import numpy as np
import pandas as pd
import pytest
pytest.importorskip('torch')
pytest.importorskip('faiss')
pytest.importorskip('lightgbm')
from recsyslab.neural import TwoTower, negative_sample
from recsyslab.ranking import TwoStage, features, ranking_data, fit_ranker, FEATURES
from recsyslab.stage3 import forward_fold, run, candidate_metrics
from recsyslab.data import temporal_split


@pytest.fixture
def data():
    rows = [(user + 1, (round_ + user) % 8 + 10, 5., round_ * 5 + user + 1)
            for round_ in range(8) for user in range(5)]
    ratings = pd.DataFrame(rows, columns=['user_id', 'movie_id', 'rating', 'timestamp'])
    movies = pd.DataFrame({'movie_id': np.arange(10, 18), 'title': list('ABCDEFGH'),
                           'genres': ['Comedy', 'Drama'] * 4})
    return ratings, movies


def fit(history, movies):
    return TwoTower.fit(history, movies, dimension=4, epochs=2, batch_size=8, seed=42, min_positives=1)


def test_negative_sampling_excludes_all_known_positives(data):
    ratings, movies = data
    model = fit(ratings.iloc[:20], movies)
    users = np.tile(np.arange(5), 100)
    negative = negative_sample(users, model.profiles, np.random.default_rng(42))
    assert np.asarray(model.profiles[users, negative]).sum() == 0
    from scipy import sparse
    with pytest.raises(ValueError): negative_sample(np.array([0]), sparse.csr_matrix([[1, 1]]), np.random.default_rng(1))


def test_faiss_bruteforce_and_seen_exclusion(data):
    ratings, movies = data
    model = fit(ratings.iloc[:20], movies)
    user = 1
    scores = model.item_vectors @ model.user_vectors[model.user_index[user]]
    seen = {10, 11, 12}
    expected = [int(model.items[i]) for i in np.lexsort((model.items, -scores)) if int(model.items[i]) not in seen][:3]
    assert model.candidates(seen, 3, user) == expected
    assert len(model.candidates(seen, 2, user)) == 2
    assert model.candidates(set(model.catalog), 3, user) == []
    assert model.candidates(seen, 3, 999) == model.baseline.recommend(seen, 3)
    with pytest.raises(ValueError): model.candidates(seen, 0, user)


def test_temporal_features_and_no_positive_injection(data):
    ratings, movies = data
    history, labels, cutoff = forward_fold(ratings.iloc[:32], .75)
    model = fit(history, movies)
    x, y, groups, audit, stats = ranking_data(model, history, labels, 8)
    assert stats['feature_history_max_timestamp'] < stats['label_min_timestamp']
    assert sum(groups) == len(y) == len(x)
    for user in audit.user_id.unique():
        seen = set(history[history.user_id == user].movie_id)
        candidates = model.candidates(seen, 8, int(user))
        assert audit[audit.user_id == user].movie_id.tolist() == candidates
        np.testing.assert_array_equal(features(model, int(user), candidates), x[audit.user_id == user])
    # Label changes cannot modify candidates or features; all-zero groups must fail, never inject hits.
    changed = labels.copy(); changed.rating = 1
    with pytest.raises(ValueError, match='no informative'): ranking_data(model, history, changed, 8)
    with pytest.raises(ValueError, match='strictly later'): ranking_data(model, history, history, 8)


def test_refit_roundtrip_schema_and_corruption(data, tmp_path):
    ratings, movies = data
    history, labels, _ = forward_fold(ratings.iloc[:32], .75)
    a, b = fit(history, movies), fit(history.sample(frac=1, random_state=2), movies)
    np.testing.assert_array_equal(a.user_vectors, b.user_vectors)
    np.testing.assert_array_equal(a.item_vectors, b.item_vectors)
    x, y, groups, _, _ = ranking_data(a, history, labels, 8)
    ranker = fit_ranker(x, y, groups, seed=42, rounds=3, min_data_in_leaf=1, num_leaves=3)
    pipeline = TwoStage(a, ranker, 8)
    pipeline.save(tmp_path / 'bundle')
    restored = TwoStage.load(tmp_path / 'bundle')
    for user in [1, 2, 999]:
        seen = set(history[history.user_id == user].movie_id)
        recs = restored.recommend(seen, 3, user)
        assert recs == pipeline.recommend(seen, 3, user)
        assert set(recs).issubset(a.candidates(seen, 8, user))
    with pytest.raises(ValueError): pipeline.recommend(set(), 9, 1)
    path = tmp_path / 'bundle/retriever/arrays.npz'
    with path.open('ab') as stream: stream.write(b'corruption')
    with pytest.raises(ValueError, match='checksum'): TwoStage.load(tmp_path / 'bundle')


def test_candidate_recall_is_oracle_bound(data):
    ratings, movies = data
    history, labels, _ = forward_fold(ratings.iloc[:32], .75)
    model = fit(history, movies)
    summary, frame = candidate_metrics(model, history, labels, 8, 2, 42, 20)
    assert summary['candidate_recall']['mean'] == 1
    assert (frame.oracle_recall_at_k <= frame.candidate_recall).all()


def test_pipeline_and_outer_holdout_isolation(data, tmp_path):
    ratings, movies = data
    ratings.rename(columns={'user_id': 'userId', 'movie_id': 'movieId'}).to_csv(tmp_path / 'ratings.csv', index=False)
    movies.rename(columns={'movie_id': 'movieId'}).to_csv(tmp_path / 'movies.csv', index=False)
    config = {'dataset': '32m', 'train_fraction': .8, 'validation_fraction': .1, 'positive_threshold': 4.,
              'k': 2, 'seed': 42, 'bootstrap_samples': 20, 'min_positives': 1, 'candidate_limit': 8,
              'retriever_fraction': .75, 'two_tower': {'dimension': 4, 'epochs': 2, 'batch_size': 8},
              'ranker': {'rounds': 3, 'min_data_in_leaf': 1, 'num_leaves': 3},
              'item_cf': {'neighbors': 3, 'shrinkage': 1.}}
    results = run(config, tmp_path, tmp_path / 'first')
    ratings.loc[ratings.timestamp.isin([33, 34, 37, 38]), 'rating'] = 1
    ratings.rename(columns={'user_id': 'userId', 'movie_id': 'movieId'}).to_csv(tmp_path / 'ratings.csv', index=False)
    run(config, tmp_path, tmp_path / 'second')
    a, b = TwoStage.load(tmp_path / 'first/two_stage'), TwoStage.load(tmp_path / 'second/two_stage')
    np.testing.assert_array_equal(a.retriever.user_vectors, b.retriever.user_vectors)
    np.testing.assert_array_equal(a.retriever.item_vectors, b.retriever.item_vectors)
    assert a.ranker.model_to_string() == b.ranker.model_to_string()
    assert set(results['test']) == {'popularity', 'item_cf', 'two_tower', 'two_stage'}
    assert json.loads((tmp_path / 'first/selection.json').read_text())['selected_model'] in results['validation']


def test_reject_retriever_trained_on_label_window(data):
    ratings, movies = data
    history, labels, _ = forward_fold(ratings.iloc[:32], .75)
    contaminated = fit(ratings.iloc[:32], movies)
    with pytest.raises(ValueError, match='retriever was trained'):
        ranking_data(contaminated, history, labels, 8)


def test_cli_in_fresh_process():
    import os
    import subprocess
    import sys
    result = subprocess.run([sys.executable, '-m', 'recsyslab.stage3', '--help'],
                            env=dict(os.environ, MPLCONFIGDIR='/tmp/recsyslab-mpl'),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--output' in result.stdout
