"""Synthetic fixtures only; all published benchmarks use official observations."""
import json
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from recsyslab.classical import ItemCF, ImplicitALS, als_step, load_model
from recsyslab.benchmark import paired_comparison, run_benchmark, METRICS
from recsyslab.models import Popularity


@pytest.fixture
def train():
    return pd.DataFrame([(1, 10, 5, 1), (1, 20, 5, 2), (2, 10, 4, 3),
                         (2, 30, 5, 4), (3, 20, 4, 5), (3, 30, 5, 6),
                         (4, 40, 2, 7)], columns=['user_id', 'movie_id', 'rating', 'timestamp'])


def test_cosine_and_self_exclusion(train):
    model = ItemCF.fit(train, neighbors=3, shrinkage=0, min_positives=1)
    expected = np.array([[0, .5, .5, 0], [.5, 0, .5, 0], [.5, .5, 0, 0], [0, 0, 0, 0]])
    np.testing.assert_allclose(model.similarity.toarray(), expected)
    assert model.recommend({10, 20}, 2, 1) == [30, 40]
    bounded = ItemCF.fit(train, neighbors=1)
    assert np.max(np.diff(bounded.similarity.indptr)) <= 1


def test_als_matches_dense_normal_equations():
    interactions = sparse.csr_matrix([[1, 0, 1], [0, 0, 0]])
    factors = np.array([[1., 2.], [2., 1.], [1., -1.]])
    confidence = np.diag([6., 1., 6.])
    expected = np.linalg.solve(factors.T @ confidence @ factors + .2 * np.eye(2),
                               factors.T @ confidence @ np.array([1, 0, 1]))
    result = als_step(interactions, factors, 5, .2)
    np.testing.assert_allclose(result[0], expected)
    np.testing.assert_array_equal(result[1], [0, 0])


@pytest.mark.parametrize('kind', [ItemCF, ImplicitALS])
def test_repeatability_persistence_and_fallback(train, tmp_path, kind):
    params = {'min_positives': 2}
    if kind is ImplicitALS: params.update(factors=3, iterations=3, seed=2)
    a, b = kind.fit(train, **params), kind.fit(train.sample(frac=1, random_state=1), **params)
    for index in range(len(a.users)):
        np.testing.assert_allclose(a.scores(index), b.scores(index), rtol=0, atol=0)
    a.save(tmp_path / 'model')
    restored = load_model(tmp_path / 'model')
    for user in [1, 2, 4, 999]:
        assert restored.recommend({10}, 10, user) == a.recommend({10}, 10, user)
    assert a.recommend({10}, 5, 999) == Popularity.fit(train).recommend({10}, 5)
    assert a.uses_fallback(4)
    assert a.recommend(set(a.catalog), 2, 1) == []
    with pytest.raises(ValueError): a.recommend(set(), 0, 1)


def test_pairing_by_id_and_mismatch():
    a = pd.DataFrame({'user_id': [1, 2], **{key: [0., .5] for key in METRICS}})
    b = pd.DataFrame({'user_id': [2, 1], **{key: [.75, .25] for key in METRICS}})
    result = paired_comparison(a, b, 42, 20)
    assert result['metrics']['recall'] == {'mean_difference': .25, 'ci95': [.25, .25]}
    with pytest.raises(ValueError): paired_comparison(a, b.iloc[:1], 42, 20)


def test_phase2_pipeline_and_heldout_isolation(tmp_path, train):
    # Separate future users/items from training; changes to target values cannot affect artifacts.
    extra = pd.DataFrame([(5, 20, 5, 8), (5, 10, 4, 9), (6, 30, 5, 10)], columns=train.columns)
    ratings = pd.concat([train, extra])
    ratings.rename(columns={'user_id': 'userId', 'movie_id': 'movieId'}).to_csv(tmp_path / 'ratings.csv', index=False)
    pd.DataFrame({'movieId': [10, 20, 30, 40], 'title': ['A', 'B', 'C', 'D'],
                  'genres': ['Comedy', 'Drama', 'Comedy', 'Drama']}).to_csv(tmp_path / 'movies.csv', index=False)
    config = {'dataset': '32m', 'train_fraction': .6, 'validation_fraction': .2,
              'positive_threshold': 4., 'k': 2, 'seed': 42, 'bootstrap_samples': 20, 'min_positives': 1,
              'item_cf': {'neighbors': 2, 'shrinkage': 1.},
              'implicit_als': {'factors': 2, 'iterations': 2, 'alpha': 5., 'regularization': .1}}
    first = run_benchmark(config, tmp_path, tmp_path / 'first')
    ratings.loc[ratings.timestamp >= 7, 'rating'] = 4
    ratings.rename(columns={'user_id': 'userId', 'movie_id': 'movieId'}).to_csv(tmp_path / 'ratings.csv', index=False)
    run_benchmark(config, tmp_path, tmp_path / 'second')
    for name in ['item_cf', 'implicit_als']:
        a, b = load_model(tmp_path / 'first' / name), load_model(tmp_path / 'second' / name)
        for index in range(len(a.users)):
            np.testing.assert_array_equal(a.scores(index), b.scores(index))
    assert set(first['test']) == {'popularity', 'item_cf', 'implicit_als'}
    assert json.loads((tmp_path / 'first/selection.json').read_text())['selected_model'] in first['test']
