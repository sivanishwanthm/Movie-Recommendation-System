"""Temporal out-of-fold LambdaRank features and bounded two-stage inference."""
import json
from pathlib import Path
from typing import Optional, Set, List
# Load the pinned PyTorch OpenMP runtime before LightGBM on macOS.
import torch
import lightgbm as lgb
import numpy as np
import pandas as pd
from .data import digest
from .neural import TwoTower

FEATURES = ['tower_score', 'reciprocal_retrieval_rank', 'log_item_positives',
            'log_user_positives', 'genre_affinity']


def features(retriever: TwoTower, user: int, candidates: List[int]) -> np.ndarray:
    indices = np.array([retriever.item_index[item] for item in candidates], dtype=np.int64)
    ui = retriever.user_index.get(user)
    if ui is None:
        scores, activity, affinity = np.zeros(len(indices)), 0, np.zeros(len(indices))
    else:
        scores = retriever.item_vectors[indices] @ retriever.user_vectors[ui]
        activity = retriever.activity[ui]
        affinity = (retriever.genres[indices] @ retriever.genre_profiles[ui]) / np.maximum(retriever.genres[indices].sum(1), 1)
    matrix = np.column_stack([scores, 1 / np.arange(1, len(indices) + 1),
                              np.log1p([retriever.counts.get(item, 0) for item in candidates]),
                              np.full(len(indices), np.log1p(activity)), affinity]).astype(np.float64)
    if matrix.shape != (len(candidates), len(FEATURES)) or not np.isfinite(matrix).all():
        raise ValueError('invalid ranker feature matrix')
    return matrix


def ranking_data(retriever: TwoTower, history: pd.DataFrame, labels: pd.DataFrame, limit: int):
    if history.empty or labels.empty or history.timestamp.max() >= labels.timestamp.min():
        raise ValueError('ranker labels must be strictly later than feature history')
    if retriever.training_max_timestamp >= labels.timestamp.min():
        raise ValueError('retriever was trained on or after the ranker label window')
    seen = history.groupby('user_id').movie_id.agg(set).to_dict()
    positives = labels[labels.rating >= retriever.threshold].groupby('user_id').movie_id.agg(set).to_dict()
    xs, ys, groups, audit = [], [], [], []
    skipped = {'fallback': 0, 'no_positive_candidate': 0, 'no_negative_candidate': 0}
    for user in sorted(labels.user_id.unique()):
        user = int(user)
        if retriever.uses_fallback(user):
            skipped['fallback'] += 1
            continue
        candidates = retriever.candidates(seen.get(user, set()), limit, user)
        # Features/candidates precede label lookup; positives are NEVER injected.
        matrix = features(retriever, user, candidates)
        target = positives.get(user, set())
        y = np.array([int(item in target) for item in candidates], dtype=np.int32)
        if not y.any():
            skipped['no_positive_candidate'] += 1
            continue
        if y.all():
            skipped['no_negative_candidate'] += 1
            continue
        xs.append(matrix); ys.append(y); groups.append(len(y))
        audit.extend((user, item, int(label)) for item, label in zip(candidates, y))
    if not groups:
        raise ValueError('no informative temporal ranker groups; need more history or candidates')
    stats = {'groups': len(groups), 'rows': sum(groups), 'positive_labels': int(sum(y.sum() for y in ys)),
             'skipped_users': skipped, 'feature_history_max_timestamp': int(history.timestamp.max()),
             'label_min_timestamp': int(labels.timestamp.min()), 'label_max_timestamp': int(labels.timestamp.max())}
    return np.vstack(xs), np.concatenate(ys), groups, pd.DataFrame(audit, columns=['user_id', 'movie_id', 'label']), stats


def fit_ranker(matrix: np.ndarray, labels: np.ndarray, groups: List[int], seed: int,
               rounds: int = 60, num_leaves: int = 15, min_data_in_leaf: int = 30):
    if sum(groups) != len(labels) or matrix.shape != (len(labels), len(FEATURES)) or rounds < 1:
        raise ValueError('invalid ranker group or feature schema')
    if not np.isfinite(matrix).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError('invalid ranker values')
    dataset = lgb.Dataset(matrix, label=labels, group=groups, feature_name=FEATURES)
    return lgb.train({'objective': 'lambdarank', 'metric': 'ndcg', 'ndcg_eval_at': [10],
                      'learning_rate': .05, 'num_leaves': num_leaves, 'min_data_in_leaf': min_data_in_leaf,
                      'num_threads': 1, 'seed': seed, 'deterministic': True, 'force_col_wise': True,
                      'verbosity': -1}, dataset, num_boost_round=rounds)


class TwoStage:
    def __init__(self, retriever: TwoTower, ranker: Optional[lgb.Booster], candidate_limit: int):
        if candidate_limit < 1:
            raise ValueError('candidate limit must be positive')
        if ranker is not None and ranker.feature_name() != FEATURES:
            raise ValueError('incompatible ranker feature schema')
        self.retriever, self.ranker, self.candidate_limit = retriever, ranker, candidate_limit
        self.catalog, self.counts, self.threshold = retriever.catalog, retriever.counts, retriever.threshold

    def recommend(self, seen: Set[int], k: int, user_id: Optional[int] = None) -> List[int]:
        if not 1 <= k <= self.candidate_limit:
            raise ValueError('K must be positive and within candidate budget')
        candidates = self.retriever.candidates(seen, self.candidate_limit, user_id)
        if self.ranker is None or self.retriever.uses_fallback(user_id) or not candidates:
            return candidates[:k]
        scores = self.ranker.predict(features(self.retriever, user_id, candidates), num_threads=1)
        if not np.isfinite(scores).all():
            raise ValueError('nonfinite ranker scores')
        order = np.lexsort((np.arange(len(candidates)), -scores))
        return [candidates[i] for i in order[:k]]

    def save(self, directory: Path) -> None:
        if self.ranker is None:
            raise ValueError('persist the complete ranked model')
        directory.mkdir(parents=True, exist_ok=False)
        self.retriever.save(directory / 'retriever')
        self.ranker.save_model(str(directory / 'ranker.txt'))
        metadata = {'schema_version': 1, 'features': FEATURES, 'candidate_limit': self.candidate_limit,
                    'ranker_sha256': digest(directory / 'ranker.txt'),
                    'retriever_bundle_sha256': digest(directory / 'retriever/checksums.json')}
        (directory / 'bundle.json').write_text(json.dumps(metadata, indent=2))

    @classmethod
    def load(cls, directory: Path):
        metadata = json.loads((directory / 'bundle.json').read_text())
        if metadata['schema_version'] != 1 or metadata['features'] != FEATURES:
            raise ValueError('incompatible two-stage schema')
        if digest(directory / 'ranker.txt') != metadata['ranker_sha256'] or digest(directory / 'retriever/checksums.json') != metadata['retriever_bundle_sha256']:
            raise ValueError('incompatible ranker/retriever bundle')
        return cls(TwoTower.load(directory / 'retriever'), lgb.Booster(model_file=str(directory / 'ranker.txt')),
                   metadata['candidate_limit'])
