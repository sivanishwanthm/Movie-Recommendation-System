"""Official MovieLens ingestion and strict relational validation."""
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def download(dataset: str, root: Path) -> Path:
    if dataset not in {'1m', '32m'}:
        raise ValueError('dataset must be 1m or 32m')
    root.mkdir(parents=True, exist_ok=True)
    url = f'https://files.grouplens.org/datasets/movielens/ml-{dataset}.zip'
    archive = root / f'ml-{dataset}.zip'
    if not archive.exists():
        temporary = archive.with_suffix('.part')
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as out:
            shutil.copyfileobj(response, out)
        temporary.replace(archive)
    # Reject traversal and corruption before extracting the official archive.
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            if not (root / member.filename).resolve().is_relative_to(root.resolve()):
                raise ValueError('unsafe archive member')
        if z.testzip() is not None:
            raise ValueError('archive CRC validation failed')
        z.extractall(root)
    (root / f'ml-{dataset}-source.json').write_text(json.dumps({
        'url': url, 'sha256': digest(archive),
        'note': 'Recorded download fingerprint; not an independently verified publisher checksum.'
    }, indent=2))
    return root / f'ml-{dataset}'


def load(directory: Path, dataset: str) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    if dataset == '1m':
        ratings = pd.read_csv(directory / 'ratings.dat', sep='::', engine='python',
                              names=['user_id', 'movie_id', 'rating', 'timestamp'])
        movies = pd.read_csv(directory / 'movies.dat', sep='::', engine='python', encoding='latin-1',
                             names=['movie_id', 'title', 'genres'])
        users = pd.read_csv(directory / 'users.dat', sep='::', engine='python',
                            names=['user_id', 'gender', 'age', 'occupation', 'zip_code'],
                            dtype={'zip_code': str})
    elif dataset == '32m':
        ratings = pd.read_csv(directory / 'ratings.csv').rename(columns={'userId': 'user_id', 'movieId': 'movie_id'})
        movies = pd.read_csv(directory / 'movies.csv').rename(columns={'movieId': 'movie_id'})
        users = None  # GroupLens does not supply demographics in 32M.
    else:
        raise ValueError('dataset must be 1m or 32m')
    validate(ratings, movies, users, dataset)
    return ratings, movies, users


def validate(ratings: pd.DataFrame, movies: pd.DataFrame,
             users: Optional[pd.DataFrame], dataset: str) -> None:
    for frame, cols in [(ratings, ['user_id', 'movie_id', 'rating', 'timestamp']),
                        (movies, ['movie_id', 'title', 'genres'])]:
        if not set(cols).issubset(frame.columns) or frame.empty or frame[cols].isna().any().any():
            raise ValueError('missing, empty, or null required data')
    for frame, cols in [(ratings, ['user_id', 'movie_id', 'timestamp']), (movies, ['movie_id'])]:
        for col in cols:
            values = frame[col].to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values <= 0).any() or (values % 1 != 0).any():
                raise ValueError(f'invalid integer {col}')
    allowed = np.arange(1, 6) if dataset == '1m' else np.arange(0.5, 5.1, 0.5)
    if not ratings.rating.isin(allowed).all():
        raise ValueError('invalid rating scale')
    if movies.movie_id.duplicated().any() or ratings.duplicated(['user_id', 'movie_id']).any():
        raise ValueError('duplicate entity or user/movie rating')
    if not ratings.movie_id.isin(movies.movie_id).all():
        raise ValueError('unknown movie reference')
    if users is not None:
        if users.empty or users.isna().any().any() or users.user_id.duplicated().any():
            raise ValueError('invalid user metadata')
        if not ratings.user_id.isin(users.user_id).all():
            raise ValueError('unknown user reference')


def temporal_split(ratings: pd.DataFrame, train_fraction: float, validation_fraction: float):
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 - train_fraction:
        raise ValueError('split fractions must be positive and sum to less than one')
    times = np.sort(ratings.timestamp.to_numpy())
    a = int(times[int(len(times) * train_fraction)])
    b = int(times[int(len(times) * (train_fraction + validation_fraction))])
    train = ratings[ratings.timestamp < a].copy()
    validation = ratings[(ratings.timestamp >= a) & (ratings.timestamp < b)].copy()
    test = ratings[ratings.timestamp >= b].copy()
    if any(part.empty for part in (train, validation, test)):
        raise ValueError('timestamp ties or insufficient history produced an empty split')
    return train, validation, test, {'validation_start': a, 'test_start': b}
