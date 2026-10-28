"""Tiny synthetic fixtures test correctness; they are never benchmark results."""
import json
import numpy as np
import pandas as pd
import pytest
from recsyslab.data import load, temporal_split, validate
from recsyslab.models import Popularity
from recsyslab.evaluation import evaluate, ranking_metrics
from recsyslab.pipeline import run


@pytest.fixture
def frames():
    ratings = pd.DataFrame([(1, 1, 5, 10), (2, 2, 4, 20), (1, 3, 2, 30),
                            (2, 3, 5, 40), (3, 1, 5, 50), (3, 2, 4, 60),
                            (4, 1, 5, 70), (4, 3, 4, 80), (5, 2, 5, 90), (5, 3, 3, 100)],
                           columns=['user_id', 'movie_id', 'rating', 'timestamp'])
    movies = pd.DataFrame([(1, 'A', 'Comedy'), (2, 'B', 'Drama'), (3, 'C', 'Comedy|Drama')],
                          columns=['movie_id', 'title', 'genres'])
    return ratings, movies


def test_temporal_ties_and_partition(frames):
    ratings, _ = frames
    ratings.loc[7, 'timestamp'] = 70
    train, val, test, _ = temporal_split(ratings, .6, .2)
    assert train.timestamp.max() < val.timestamp.min() <= val.timestamp.max() < test.timestamp.min()
    assert len(train) + len(val) + len(test) == len(ratings)
    assert set(train.index).isdisjoint(test.index)


def test_empty_split_fails(frames):
    ratings, _ = frames
    ratings.timestamp = 10
    with pytest.raises(ValueError):
        temporal_split(ratings, .8, .1)


@pytest.mark.parametrize('issue', ['rating', 'duplicate', 'reference', 'timestamp', 'null'])
def test_validation_rejects_bad_data(frames, issue):
    ratings, movies = frames
    if issue == 'rating': ratings.loc[0, 'rating'] = 6
    if issue == 'duplicate': ratings = pd.concat([ratings, ratings.iloc[:1]])
    if issue == 'reference': ratings.loc[0, 'movie_id'] = 999
    if issue == 'timestamp':
        ratings['timestamp'] = ratings.timestamp.astype(float)
        ratings.loc[0, 'timestamp'] = 1.5
    if issue == 'null': movies.loc[0, 'title'] = None
    with pytest.raises(ValueError):
        validate(ratings, movies, None, '1m')


def test_metric_arithmetic():
    result = ranking_metrics([1, 2, 3], {1, 3, 9}, 3)
    assert result['recall'] == pytest.approx(2 / 3)
    assert result['precision'] == pytest.approx(2 / 3)
    assert result['ndcg'] == pytest.approx(1.5 / (1 + 1 / np.log2(3) + .5))
    assert ranking_metrics([], {9}, 10)['hit_rate'] == 0


def test_popularity_seen_ties_and_no_future(frames):
    ratings, _ = frames
    train = ratings.iloc[:3]
    model = Popularity.fit(train)
    assert model.recommend(set(), 10) == [1, 2, 3]
    assert model.recommend({1, 2}, 10) == [3]
    ratings.loc[3:, 'rating'] = 1
    assert Popularity.fit(train).counts == model.counts


def test_cold_items_users_and_determinism(frames):
    ratings, movies = frames
    train = ratings.iloc[:2]
    model = Popularity.fit(train)
    target = pd.DataFrame([(99, 3, 5, 200)], columns=ratings.columns)
    a, rows = evaluate(model, train, target, movies, 2, 42, 50)
    b, _ = evaluate(model, train, target, movies, 2, 42, 50)
    assert a['metrics'] == b['metrics']
    assert a['cold_start_users'] == 1
    assert a['unreachable_positive_fraction'] == 1
    assert a['metrics']['recall']['mean'] == 0
    assert rows.reachable_positives.iloc[0] == 0


def test_32m_loader_and_pipeline(tmp_path, frames):
    ratings, movies = frames
    ratings.rename(columns={'user_id': 'userId', 'movie_id': 'movieId'}).to_csv(tmp_path / 'ratings.csv', index=False)
    movies.rename(columns={'movie_id': 'movieId'}).to_csv(tmp_path / 'movies.csv', index=False)
    assert load(tmp_path, '32m')[2] is None
    config = {'dataset': '32m', 'train_fraction': .6, 'validation_fraction': .2,
              'positive_threshold': 4, 'k': 2, 'seed': 42, 'bootstrap_samples': 20}
    result = run(config, tmp_path, tmp_path / 'output')
    assert result['test']['users_evaluated'] == 1
    manifest = json.loads((tmp_path / 'output/manifest.json').read_text())
    assert manifest['rows'] == {'train': 6, 'validation': 2, 'test': 2}
    assert (tmp_path / 'output/eda.png').exists()
    with pytest.raises(FileExistsError):
        run(config, tmp_path, tmp_path / 'output')


def test_1m_loader_preserves_metadata(tmp_path):
    (tmp_path / 'ratings.dat').write_text('1::1::5::100\n')
    (tmp_path / 'movies.dat').write_bytes('1::Amélie::Comedy\n'.encode('latin-1'))
    (tmp_path / 'users.dat').write_text('1::F::25::1::00123\n')
    ratings, movies, users = load(tmp_path, '1m')
    assert users.zip_code.iloc[0] == '00123'
    assert movies.title.iloc[0] == 'Amélie'
