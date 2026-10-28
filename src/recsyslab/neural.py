"""CPU two-tower training and exact FAISS inner-product candidate retrieval."""
import json
from pathlib import Path
from typing import Optional, Set, List
import faiss
import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch import nn
from torch.nn import functional as F
from .classical import positive_matrix
from .data import digest
from .models import Popularity


class Towers(nn.Module):
    def __init__(self, users: int, items: int, genres: np.ndarray, dimension: int):
        super().__init__()
        self.user_embedding = nn.Embedding(users, dimension)
        self.item_embedding = nn.Embedding(items, dimension)
        self.register_buffer('genres', torch.tensor(genres, dtype=torch.float32))
        self.user_mlp = nn.Sequential(nn.Linear(dimension, dimension * 2), nn.ReLU(),
                                      nn.Linear(dimension * 2, dimension))
        self.item_mlp = nn.Sequential(nn.Linear(dimension + genres.shape[1], dimension * 2), nn.ReLU(),
                                      nn.Linear(dimension * 2, dimension))

    def user(self, indices: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.user_mlp(self.user_embedding(indices)), dim=1)

    def item(self, indices: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.item_mlp(torch.cat([self.item_embedding(indices), self.genres[indices]], dim=1)), dim=1)


def negative_sample(users: np.ndarray, positives: sparse.csr_matrix, rng: np.random.Generator) -> np.ndarray:
    """Uniform catalog draws excluding every positive in the training partition."""
    if np.any(np.diff(positives.indptr)[users] >= positives.shape[1]):
        raise ValueError('user has no available negative item')
    negatives = rng.integers(0, positives.shape[1], len(users))
    invalid = np.asarray(positives[users, negatives]).ravel() > 0
    while invalid.any():
        negatives[invalid] = rng.integers(0, positives.shape[1], invalid.sum())
        invalid = np.asarray(positives[users, negatives]).ravel() > 0
    return negatives


class TwoTower:
    def __init__(self, baseline, users, items, profiles, genres, params):
        self.baseline = baseline
        self.catalog, self.counts, self.threshold = baseline.catalog, baseline.counts, baseline.threshold
        self.users, self.items, self.profiles, self.genres, self.params = users, items, profiles, genres, params
        self.user_index = {int(user): index for index, user in enumerate(users)}
        self.item_index = {int(item): index for index, item in enumerate(items)}
        self.activity = np.diff(profiles.indptr)
        self.genre_profiles = np.asarray(profiles @ genres) / np.maximum(self.activity[:, None], 1)

    @classmethod
    def fit(cls, train: pd.DataFrame, movies: pd.DataFrame, threshold: float = 4., dimension: int = 32,
            epochs: int = 5, batch_size: int = 2048, learning_rate: float = .002,
            temperature: float = .2, min_positives: int = 3, seed: int = 42):
        if min(dimension, epochs, batch_size, min_positives) < 1 or learning_rate <= 0 or temperature <= 0:
            raise ValueError('invalid two-tower configuration')
        if not np.isfinite([threshold, learning_rate, temperature]).all():
            raise ValueError('nonfinite two-tower configuration')
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(seed)
        faiss.omp_set_num_threads(1)
        users, items, profiles = positive_matrix(train, threshold)
        metadata = movies.set_index('movie_id').loc[items]
        vocab = sorted({g for value in metadata.genres for g in value.split('|')} - {'(no genres listed)'})
        genre_index = {g: j for j, g in enumerate(vocab)}
        genres = np.zeros((len(items), len(vocab)), dtype=np.float32)
        for i, value in enumerate(metadata.genres):
            for genre in value.split('|'):
                if genre in genre_index:
                    genres[i, genre_index[genre]] = 1
        params = dict(dimension=dimension, epochs=epochs, batch_size=batch_size,
                      learning_rate=learning_rate, temperature=temperature, min_positives=min_positives, seed=seed)
        model = cls(Popularity.fit(train, threshold), users, items, profiles, genres, params)
        model.training_max_timestamp = int(train.timestamp.max())
        model.genre_vocabulary = vocab
        model.network = Towers(len(users), len(items), genres, dimension)
        optimizer = torch.optim.Adam(model.network.parameters(), lr=learning_rate, weight_decay=1e-5)
        row, col = profiles.nonzero()
        eligible = model.activity[row] < len(items)
        row, col = row[eligible], col[eligible]
        if len(row) == 0:
            raise ValueError('no positive pairs with available training negatives')
        rng = np.random.default_rng(seed)
        model.losses = []
        for epoch in range(epochs):
            permutation = rng.permutation(len(row))
            total = 0.
            for start in range(0, len(row), batch_size):
                batch = permutation[start:start + batch_size]
                u, pos = row[batch], col[batch]
                neg = negative_sample(u, profiles, rng)
                uv = model.network.user(torch.tensor(u, dtype=torch.long))
                pv = model.network.item(torch.tensor(pos, dtype=torch.long))
                nv = model.network.item(torch.tensor(neg, dtype=torch.long))
                loss = F.softplus(-((uv * pv).sum(1) - (uv * nv).sum(1)) / temperature).mean()
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite training loss')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item() * len(batch)
            model.losses.append(total / len(row))
            print(f'two-tower epoch {epoch + 1}/{epochs}: loss={model.losses[-1]:.6f}', flush=True)
        model.network.eval()
        with torch.no_grad():
            model.user_vectors = model.network.user(torch.arange(len(users))).numpy()
            model.item_vectors = model.network.item(torch.arange(len(items))).numpy()
        model.index = faiss.IndexFlatIP(dimension)
        model.index.add(np.ascontiguousarray(model.item_vectors))
        return model

    def uses_fallback(self, user_id: Optional[int]) -> bool:
        index = self.user_index.get(user_id)
        return index is None or self.activity[index] < self.params['min_positives']

    def candidates(self, seen: Set[int], limit: int, user_id: Optional[int] = None) -> List[int]:
        if limit < 1:
            raise ValueError('candidate limit must be positive')
        if self.uses_fallback(user_id):
            return self.baseline.recommend(seen, limit)
        # Over-fetch just enough to replace every possible seen result; final output <= limit.
        fetch = min(len(self.items), limit + sum(item in self.item_index for item in seen))
        query = self.user_vectors[self.user_index[user_id]:self.user_index[user_id] + 1]
        scores, indices = self.index.search(np.ascontiguousarray(query), fetch)
        order = np.lexsort((self.items[indices[0]], -scores[0]))
        return [int(self.items[indices[0, j]]) for j in order
                if int(self.items[indices[0, j]]) not in seen][:limit]

    def recommend(self, seen: Set[int], k: int, user_id: Optional[int] = None) -> List[int]:
        return self.candidates(seen, k, user_id)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=False)
        metadata = dict(schema_version=1, params=self.params, threshold=self.threshold,
                        catalog=self.catalog, counts=self.counts, genre_vocabulary=self.genre_vocabulary, losses=self.losses,
                        training_max_timestamp=self.training_max_timestamp)
        (directory / 'model.json').write_text(json.dumps(metadata, indent=2))
        np.savez_compressed(directory / 'arrays.npz', users=self.users, items=self.items, genres=self.genres,
                            user_vectors=self.user_vectors, item_vectors=self.item_vectors)
        # Pure numeric state dictionary; no pickle or executable model object.
        np.savez_compressed(directory / 'weights.npz', **{k: v.cpu().numpy() for k, v in self.network.state_dict().items()})
        sparse.save_npz(directory / 'profiles.npz', self.profiles)
        faiss.write_index(self.index, str(directory / 'index.faiss'))
        (directory / 'checksums.json').write_text(json.dumps({p.name: digest(p) for p in directory.iterdir()}, indent=2))

    @classmethod
    def load(cls, directory: Path):
        checksums = json.loads((directory / 'checksums.json').read_text())
        if set(checksums) != {'model.json', 'arrays.npz', 'weights.npz', 'profiles.npz', 'index.faiss'}:
            raise ValueError('invalid retriever bundle file schema')
        if any(digest(directory / name) != expected for name, expected in checksums.items()):
            raise ValueError('retriever artifact checksum mismatch')
        metadata = json.loads((directory / 'model.json').read_text())
        if metadata['schema_version'] != 1:
            raise ValueError('unsupported retriever version')
        baseline = Popularity(metadata['catalog'], {int(k): v for k, v in metadata['counts'].items()}, metadata['threshold'])
        with np.load(directory / 'arrays.npz', allow_pickle=False) as arrays:
            model = cls(baseline, arrays['users'], arrays['items'], sparse.load_npz(directory / 'profiles.npz'),
                        arrays['genres'], metadata['params'])
            model.user_vectors, model.item_vectors = arrays['user_vectors'], arrays['item_vectors']
        model.network = Towers(len(model.users), len(model.items), model.genres, model.params['dimension'])
        with np.load(directory / 'weights.npz', allow_pickle=False) as weights:
            model.network.load_state_dict({k: torch.from_numpy(weights[k].copy()) for k in weights.files}, strict=True)
        model.network.eval()
        model.index = faiss.read_index(str(directory / 'index.faiss'))
        if model.index.ntotal != len(model.items) or model.index.d != model.params['dimension']:
            raise ValueError('FAISS dimensions do not match retriever')
        with torch.no_grad():
            np.testing.assert_allclose(model.network.item(torch.arange(len(model.items))).numpy(), model.item_vectors, atol=1e-6)
            np.testing.assert_allclose(model.network.user(torch.arange(len(model.users))).numpy(), model.user_vectors, atol=1e-6)
        np.testing.assert_array_equal(model.index.reconstruct_n(0, len(model.items)), model.item_vectors)
        model.training_max_timestamp = metadata['training_max_timestamp']
        model.genre_vocabulary, model.losses = metadata['genre_vocabulary'], metadata['losses']
        return model
