"""Train-only binary-positive item CF and confidence-weighted implicit ALS."""
from pathlib import Path
from typing import Optional, Set, List
import json
import numpy as np
import pandas as pd
from scipy import sparse
from .models import Popularity


def positive_matrix(train: pd.DataFrame, threshold: float):
    users = np.sort(train.user_id.unique()).astype(np.int64)
    items = np.sort(train.movie_id.unique()).astype(np.int64)
    positives = train[train.rating >= threshold]
    matrix = sparse.csr_matrix((np.ones(len(positives)),
        (np.searchsorted(users, positives.user_id), np.searchsorted(items, positives.movie_id))),
        shape=(len(users), len(items)))
    matrix.data[:] = 1.0
    return users, items, matrix


def als_step(interactions: sparse.csr_matrix, fixed: np.ndarray,
             alpha: float, regularization: float) -> np.ndarray:
    """Exact normal equations for c=1+alpha*p, including all unobserved pairs."""
    gram = fixed.T @ fixed + regularization * np.eye(fixed.shape[1])
    result = np.zeros((interactions.shape[0], fixed.shape[1]))
    for row in range(interactions.shape[0]):
        indices = interactions.indices[interactions.indptr[row]:interactions.indptr[row + 1]]
        selected = fixed[indices]
        if len(indices):
            result[row] = np.linalg.solve(gram + alpha * selected.T @ selected,
                                          (1 + alpha) * selected.sum(axis=0))
    return result


class Classical:
    """Shared deterministic ranking, sparse profiles and non-pickle persistence."""
    def __init__(self, baseline: Popularity, users: np.ndarray, items: np.ndarray,
                 profiles: sparse.csr_matrix, min_positives: int):
        self.baseline = baseline
        self.catalog, self.counts, self.threshold = baseline.catalog, baseline.counts, baseline.threshold
        self.users, self.items, self.profiles = users, items, profiles
        self.min_positives = min_positives
        self.user_index = {int(u): i for i, u in enumerate(users)}
        self.popularity = np.array([self.counts.get(int(i), 0) for i in items])

    def uses_fallback(self, user_id: Optional[int]) -> bool:
        index = self.user_index.get(user_id)
        return index is None or self.profiles.indptr[index + 1] - self.profiles.indptr[index] < self.min_positives

    def recommend(self, seen: Set[int], k: int, user_id: Optional[int] = None) -> List[int]:
        if k <= 0:
            raise ValueError('k must be positive')
        if self.uses_fallback(user_id):
            return self.baseline.recommend(seen, k)
        scores = self.scores(self.user_index[user_id])
        if not np.isfinite(scores).all():
            raise ValueError('nonfinite model scores')
        order = np.lexsort((self.items, -self.popularity, -scores))
        return [int(self.items[i]) for i in order if int(self.items[i]) not in seen][:k]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=False)
        metadata = {'schema_version': 1, 'kind': self.kind, 'params': self.params,
                    'threshold': self.threshold, 'min_positives': self.min_positives,
                    'catalog': self.catalog, 'counts': self.counts}
        (directory / 'model.json').write_text(json.dumps(metadata, indent=2))
        sparse.save_npz(directory / 'profiles.npz', self.profiles)
        arrays = {'users': self.users, 'items': self.items}
        if self.kind == 'item_cf':
            sparse.save_npz(directory / 'similarity.npz', self.similarity)
        else:
            arrays.update(user_factors=self.user_factors, item_factors=self.item_factors)
        np.savez_compressed(directory / 'arrays.npz', **arrays)


class ItemCF(Classical):
    kind = 'item_cf'

    @classmethod
    def fit(cls, train: pd.DataFrame, threshold: float = 4.0, neighbors: int = 100,
            shrinkage: float = 10.0, min_positives: int = 3):
        if neighbors < 1 or shrinkage < 0 or not np.isfinite(shrinkage) or min_positives < 1:
            raise ValueError('invalid item CF parameters')
        users, items, matrix = positive_matrix(train, threshold)
        model = cls(Popularity.fit(train, threshold), users, items, matrix, min_positives)
        model.params = {'neighbors': neighbors, 'shrinkage': shrinkage}
        norms = np.sqrt(np.asarray(matrix.sum(axis=0)).ravel())
        rows, cols, values = [], [], []
        # Compute blocks to avoid a dense item-by-item allocation.
        for start in range(0, len(items), 256):
            cooccurrence = (matrix[:, start:start + 256].T @ matrix).tocsr()
            for local in range(cooccurrence.shape[0]):
                item = start + local
                lo, hi = cooccurrence.indptr[local:local + 2]
                idx, count = cooccurrence.indices[lo:hi], cooccurrence.data[lo:hi]
                keep = idx != item
                idx, count = idx[keep], count[keep]
                score = count / (norms[item] * norms[idx]) * count / (count + shrinkage)
                best = np.lexsort((items[idx], -score))[:neighbors]
                rows.extend([item] * len(best)); cols.extend(idx[best]); values.extend(score[best])
        model.similarity = sparse.csr_matrix((values, (rows, cols)), shape=(len(items), len(items)))
        return model

    def scores(self, user_index: int) -> np.ndarray:
        return (self.profiles[user_index] @ self.similarity).toarray().ravel()


class ImplicitALS(Classical):
    kind = 'implicit_als'

    @classmethod
    def fit(cls, train: pd.DataFrame, threshold: float = 4.0, factors: int = 32,
            iterations: int = 12, alpha: float = 20.0, regularization: float = 0.1,
            seed: int = 42, min_positives: int = 3):
        if factors < 1 or iterations < 1 or alpha < 0 or regularization <= 0 or min_positives < 1:
            raise ValueError('invalid ALS parameters')
        if not np.isfinite([alpha, regularization]).all():
            raise ValueError('nonfinite ALS parameters')
        users, items, matrix = positive_matrix(train, threshold)
        model = cls(Popularity.fit(train, threshold), users, items, matrix, min_positives)
        model.params = dict(factors=factors, iterations=iterations, alpha=alpha,
                            regularization=regularization, seed=seed)
        rng = np.random.default_rng(seed)
        model.item_factors = rng.normal(0, .01, (len(items), factors))
        transpose = matrix.T.tocsr()
        for _ in range(iterations):
            model.user_factors = als_step(matrix, model.item_factors, alpha, regularization)
            model.item_factors = als_step(transpose, model.user_factors, alpha, regularization)
        return model

    def scores(self, user_index: int) -> np.ndarray:
        return self.item_factors @ self.user_factors[user_index]


def load_model(directory: Path) -> Classical:
    metadata = json.loads((directory / 'model.json').read_text())
    if metadata['schema_version'] != 1 or metadata['kind'] not in {'item_cf', 'implicit_als'}:
        raise ValueError('unsupported model artifact')
    baseline = Popularity(metadata['catalog'], {int(k): v for k, v in metadata['counts'].items()}, metadata['threshold'])
    with np.load(directory / 'arrays.npz', allow_pickle=False) as arrays:
        kind = ItemCF if metadata['kind'] == 'item_cf' else ImplicitALS
        model = kind(baseline, arrays['users'], arrays['items'],
                     sparse.load_npz(directory / 'profiles.npz'), metadata['min_positives'])
        model.params = metadata['params']
        if metadata['kind'] == 'item_cf':
            model.similarity = sparse.load_npz(directory / 'similarity.npz')
        else:
            model.user_factors, model.item_factors = arrays['user_factors'], arrays['item_factors']
    return model
