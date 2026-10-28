"""Leakage-safe two-tower retrieval / LambdaRank offline benchmark."""
import argparse
import json
import os
import platform
import time
from pathlib import Path
import faiss
# Load the pinned PyTorch OpenMP runtime before LightGBM on macOS.
import torch
import lightgbm as lgb
import numpy as np
import pandas as pd
from .benchmark import paired_comparison, METRICS
from .classical import ItemCF
from .data import load, temporal_split, digest
from .evaluation import evaluate
from .models import Popularity
from .neural import TwoTower
from .pipeline import write
from .ranking import TwoStage, ranking_data, fit_ranker, FEATURES


def forward_fold(train: pd.DataFrame, fraction: float):
    if not 0 < fraction < 1:
        raise ValueError('retriever fraction must be between zero and one')
    cutoff = int(np.sort(train.timestamp.to_numpy())[int(len(train) * fraction)])
    prefix, labels = train[train.timestamp < cutoff].copy(), train[train.timestamp >= cutoff].copy()
    if prefix.empty or labels.empty:
        raise ValueError('empty forward temporal fold')
    return prefix, labels, cutoff


def candidate_metrics(retriever, history, target, limit, k, seed, samples):
    seen = history.groupby('user_id').movie_id.agg(set).to_dict()
    positives = target[target.rating >= retriever.threshold].groupby('user_id').movie_id.agg(set)
    rows = []
    for user, relevant in positives.items():
        relevant = relevant - seen.get(user, set())
        if not relevant:
            continue
        candidates = retriever.candidates(seen.get(user, set()), limit, int(user))
        hits = len(set(candidates) & relevant)
        rows.append({'user_id': int(user), 'candidate_recall': hits / len(relevant),
                     'oracle_recall_at_k': min(hits, k) / len(relevant),
                     'candidate_count': len(candidates), 'fallback_eligible': retriever.uses_fallback(int(user))})
    frame = pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    result = {'candidate_limit': limit, 'users': len(frame), 'mean_candidates': float(frame.candidate_count.mean())}
    for metric in ['candidate_recall', 'oracle_recall_at_k']:
        values = frame[metric].to_numpy()
        boot = [rng.choice(values, len(values), replace=True).mean() for _ in range(samples)]
        result[metric] = {'mean': float(values.mean()), 'ci95': np.quantile(boot, [.025, .975]).tolist()}
    result['cohorts'] = {name: {'users': int(mask.sum()), 'candidate_recall':
                        float(frame.loc[mask, 'candidate_recall'].mean()) if mask.any() else None}
                        for name, mask in [('warm', ~frame.fallback_eligible), ('fallback_eligible', frame.fallback_eligible)]}
    return result, frame


def run(config: dict, directory: Path, output: Path):
    if not 1 <= config['k'] <= config['candidate_limit'] or config['bootstrap_samples'] < 1:
        raise ValueError('invalid K, candidate limit or bootstrap samples')
    if not .5 <= config['positive_threshold'] <= 5:
        raise ValueError('invalid positive threshold')
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); faiss.omp_set_num_threads(1)
    start = time.perf_counter()
    ratings, movies, _ = load(directory, config['dataset'])
    train, validation, test, cutoffs = temporal_split(ratings, config['train_fraction'], config['validation_fraction'])
    prefix, labels, fold_cutoff = forward_fold(train, config['retriever_fraction'])
    neural_params = dict(config['two_tower'], threshold=config['positive_threshold'],
                         min_positives=config['min_positives'], seed=config['seed'])
    times = {}
    print('Fitting forward-fold retriever', flush=True)
    tick = time.perf_counter()
    early = TwoTower.fit(prefix, movies, **neural_params)
    times['fold_retriever'] = time.perf_counter() - tick
    early.save(output / 'fold_retriever')
    matrix, y, groups, audit, fold_stats = ranking_data(early, prefix, labels, config['candidate_limit'])
    audit.to_csv(output / 'ranker_training_candidates.csv', index=False)
    np.savez_compressed(output / 'ranker_training.npz', features=matrix, labels=y, groups=groups)
    write(output / 'ranker_fold.json', dict(fold_stats, cutoff=fold_cutoff, prefix_rows=len(prefix),
          label_rows=len(labels), feature_names=FEATURES, retrieval_losses=early.losses))
    print(f'Fitting ranker: {len(groups)} groups, {len(y)} rows', flush=True)
    tick = time.perf_counter()
    ranker = fit_ranker(matrix, y, groups, config['seed'], **config['ranker'])
    times['ranker'] = time.perf_counter() - tick
    print('Refitting final retriever on outer train only', flush=True)
    tick = time.perf_counter()
    retriever = TwoTower.fit(train, movies, **neural_params)
    times['final_retriever'] = time.perf_counter() - tick
    ranked = TwoStage(retriever, ranker, config['candidate_limit'])
    ranked.save(output / 'two_stage')
    restored = TwoStage.load(output / 'two_stage')
    # Audit actual inference after loading for every fitted user and an unknown user.
    seen_train = train.groupby('user_id').movie_id.agg(set).to_dict()
    for user in list(map(int, retriever.users)) + [-1]:
        prior = seen_train.get(user, set())
        if restored.recommend(prior, config['k'], user) != ranked.recommend(prior, config['k'], user):
            raise ValueError('artifact reload changed recommendations')
    tick = time.perf_counter()
    baseline = Popularity.fit(train, config['positive_threshold'])
    times['popularity'] = time.perf_counter() - tick
    tick = time.perf_counter()
    cf = ItemCF.fit(train, threshold=config['positive_threshold'], min_positives=config['min_positives'], **config['item_cf'])
    times['item_cf'] = time.perf_counter() - tick
    models = {'popularity': baseline, 'item_cf': cf,
              'two_tower': TwoStage(retriever, None, config['candidate_limit']), 'two_stage': ranked}
    metrics, comparisons, retrieval = {}, {}, {}
    for split, target, history in [('validation', validation, train), ('test', test, pd.concat([train, validation]))]:
        metrics[split], frames, comparisons[split] = {}, {}, {}
        retrieval[split], candidates = candidate_metrics(retriever, history, target, config['candidate_limit'], config['k'], config['seed'], config['bootstrap_samples'])
        candidates.to_csv(output / f'{split}_candidate_metrics.csv', index=False)
        for name, model in models.items():
            print(f'Evaluating {split}/{name}', flush=True)
            summary, frame = evaluate(model, history, target, movies, config['k'], config['seed'], config['bootstrap_samples'])
            frame['fallback_eligible'] = frame.user_id.map(retriever.uses_fallback)
            summary['cohorts'] = {}
            for cohort, mask in [('warm', ~frame.fallback_eligible), ('fallback_eligible', frame.fallback_eligible)]:
                summary['cohorts'][cohort] = {'users': int(mask.sum()), 'metrics':
                    {m: float(frame.loc[mask, m].mean()) if mask.any() else None for m in METRICS}}
            metrics[split][name], frames[name] = summary, frame
            frame.to_csv(output / f'{split}_{name}_per_user.csv', index=False)
        for control, treatment in [('two_tower', 'two_stage'), ('item_cf', 'two_stage'), ('popularity', 'two_tower')]:
            comparisons[split][f'{treatment}_vs_{control}'] = paired_comparison(frames[control], frames[treatment], config['seed'], config['bootstrap_samples'])
        if split == 'validation':
            write(output / 'selection.json', {'selected_model': max(metrics[split], key=lambda n: metrics[split][n]['metrics']['ndcg']['mean']),
                 'rule': 'highest validation macro NDCG; fixed configurations, no test tuning'})
    write(output / 'metrics.json', metrics)
    write(output / 'paired_comparisons.json', comparisons)
    write(output / 'candidate_metrics.json', retrieval)
    write(output / 'training.json', {'fold_losses': early.losses, 'final_losses': retriever.losses,
          'ranker_feature_gain': dict(zip(FEATURES, map(float, ranker.feature_importance(importance_type='gain'))))})
    write(output / 'manifest.json', {'schema_version': 1, 'config': config, 'outer_cutoffs': cutoffs,
          'fold_cutoff': fold_cutoff, 'rows': {'train': len(train), 'validation': len(validation), 'test': len(test)},
          'fit_seconds': times, 'elapsed_seconds': time.perf_counter() - start,
          'bundle_bytes': sum(p.stat().st_size for p in (output / 'two_stage').rglob('*') if p.is_file()),
          'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__,
                      'faiss': faiss.__version__, 'lightgbm': lgb.__version__},
          'thread_environment': {k: os.environ.get(k) for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']},
          'source_sha256': {p.name: digest(p) for p in sorted(directory.iterdir()) if p.is_file()},
          'code_sha256': {p.name: digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
          'artifact_sha256': {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/phase3.json'))
    parser.add_argument('--data', type=Path, default=Path('data/raw/ml-1m'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.data, args.output)
    print(f'Phase 3 benchmark saved to {args.output}')


if __name__ == '__main__':
    main()
