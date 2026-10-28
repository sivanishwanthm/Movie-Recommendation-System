"""Full training-catalog ranking evaluation; no sampled negatives."""
import itertools
import time
from typing import Dict
import numpy as np
import pandas as pd
from .models import Recommender


def ranking_metrics(recommended, relevant, k: int) -> Dict[str, float]:
    if k <= 0 or not relevant:
        raise ValueError('positive k and nonempty relevant set required')
    hits = np.array([int(i in relevant) for i in recommended[:k]])
    dcg = float(np.sum(hits / np.log2(np.arange(len(hits)) + 2)))
    ideal = float(np.sum(1 / np.log2(np.arange(min(k, len(relevant))) + 2)))
    return {'recall': float(hits.sum() / len(relevant)), 'precision': float(hits.sum() / k),
            'ndcg': dcg / ideal, 'hit_rate': float(hits.any())}


def evaluate(model: Recommender, history: pd.DataFrame, target: pd.DataFrame,
             movies: pd.DataFrame, k: int, seed: int, bootstrap_samples: int):
    seen = history.groupby('user_id').movie_id.agg(set).to_dict()
    positives = target[target.rating >= model.threshold].groupby('user_id').movie_id.agg(set)
    catalog = set(model.catalog)
    genres = {int(r.movie_id): set(r.genres.split('|')) - {'(no genres listed)'}
              for r in movies.itertuples()}
    rows, latencies, all_recs = [], [], set()
    excluded, unreachable, total_positive = 0, 0, 0
    denominator = sum(model.counts.values()) + len(catalog)
    for user, relevant in positives.items():
        prior = seen.get(user, set())
        total_positive += len(relevant)
        unreachable += len(relevant - catalog)
        eligible = (relevant & catalog) - prior
        # Keep unreachable positives in denominator; never remove difficult users.
        relevant = relevant - prior
        if not relevant:
            excluded += 1
            continue
        start = time.perf_counter()
        recs = model.recommend(prior, k, user_id=int(user))
        latencies.append((time.perf_counter() - start) * 1000)
        all_recs.update(recs)
        row = {'user_id': int(user), 'cold_start': user not in seen,
               'reachable_positives': len(eligible), **ranking_metrics(recs, relevant, k)}
        pairs = list(itertools.combinations(recs, 2))
        row['diversity'] = float(np.mean([1 - len(genres[a] & genres[b]) / max(1, len(genres[a] | genres[b]))
                                         for a, b in pairs])) if pairs else 0.0
        row['popularity_bias'] = float(np.mean([model.counts.get(i, 0) for i in recs])) if recs else 0.0
        row['novelty_bits'] = float(np.mean([-np.log2((model.counts.get(i, 0) + 1) / denominator)
                                               for i in recs])) if recs else 0.0
        rows.append(row)
    if not rows:
        raise ValueError('no evaluable positive users')
    frame = pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    metrics = {}
    for name in ['recall', 'precision', 'ndcg', 'hit_rate', 'diversity', 'popularity_bias', 'novelty_bits']:
        values = frame[name].to_numpy()
        means = [float(rng.choice(values, len(values), replace=True).mean()) for _ in range(bootstrap_samples)]
        metrics[name] = {'mean': float(values.mean()), 'ci95': np.quantile(means, [0.025, 0.975]).tolist()}
    return {'metrics': metrics, 'users_evaluated': len(rows), 'target_users': int(target.user_id.nunique()),
            'users_without_positive': int(target.user_id.nunique() - len(positives)),
            'users_excluded_seen_only': excluded, 'cold_start_users': int(frame.cold_start.sum()),
            'unreachable_positive_fraction': unreachable / max(1, total_positive),
            'catalog_coverage': len(all_recs) / len(catalog),
            'latency_ms_p50': float(np.median(latencies)), 'latency_ms_p95': float(np.quantile(latencies, .95)),
            'recommendations_per_second': len(rows) / (sum(latencies) / 1000),
            'timing_scope': 'in-process recommendation only, sequential, no network or API'}, frame
