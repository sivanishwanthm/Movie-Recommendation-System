"""Phase 2 common-protocol benchmark and paired user-bootstrap comparisons."""
import argparse
import json
import os
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from .classical import ItemCF, ImplicitALS, load_model
from .data import load, temporal_split, digest
from .evaluation import evaluate
from .models import Popularity
from .pipeline import write

METRICS = ['recall', 'precision', 'ndcg', 'hit_rate', 'diversity', 'popularity_bias', 'novelty_bits']


def paired_comparison(control: pd.DataFrame, treatment: pd.DataFrame, seed: int, samples: int) -> dict:
    if samples < 1 or control.user_id.duplicated().any() or treatment.user_id.duplicated().any():
        raise ValueError('positive bootstrap count and unique user IDs required')
    a, b = control.set_index('user_id').sort_index(), treatment.set_index('user_id').sort_index()
    if a.empty or not a.index.equals(b.index):
        raise ValueError('paired comparisons require identical nonempty user cohorts')
    rng = np.random.default_rng(seed)
    result = {}
    for metric in METRICS:
        delta = (b[metric] - a[metric]).to_numpy()
        means = [rng.choice(delta, len(delta), replace=True).mean() for _ in range(samples)]
        result[metric] = {'mean_difference': float(delta.mean()),
                          'ci95': np.quantile(means, [.025, .975]).tolist()}
    return {'users': len(a), 'direction': 'treatment minus control', 'metrics': result}


def run_benchmark(config: dict, directory: Path, output: Path) -> dict:
    if config['k'] < 1 or config['bootstrap_samples'] < 1 or not .5 <= config['positive_threshold'] <= 5:
        raise ValueError('invalid evaluation configuration')
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    ratings, movies, _ = load(directory, config['dataset'])
    train, validation, test, cutoffs = temporal_split(ratings, config['train_fraction'], config['validation_fraction'])
    training_counts = train[train.rating >= config['positive_threshold']].groupby('user_id').size()
    threshold = config['min_positives']
    models, fit_times, artifacts = {}, {}, {}
    for name, constructor, params in [
        ('popularity', Popularity, {}),
        ('item_cf', ItemCF, dict(config['item_cf'], min_positives=threshold)),
        ('implicit_als', ImplicitALS, dict(config['implicit_als'], min_positives=threshold, seed=config['seed']))
    ]:
        print(f'Fitting {name}', flush=True)
        start = time.perf_counter()
        model = constructor.fit(train, threshold=config['positive_threshold'], **params)
        fit_times[name] = time.perf_counter() - start
        models[name] = model
        if name == 'popularity':
            write(output / 'popularity.json', {'schema_version': 1, 'catalog': model.catalog,
                  'counts': model.counts, 'threshold': model.threshold})
            artifacts[name] = (output / 'popularity.json').stat().st_size
        else:
            model.save(output / name)
            restored = load_model(output / name)
            # Check every user's raw score vector plus unknown-user fallback after loading.
            for index, user in enumerate(model.users):
                np.testing.assert_allclose(model.scores(index), restored.scores(index), rtol=0, atol=0)
            assert restored.recommend(set(), config['k'], -1) == model.recommend(set(), config['k'], -1)
            artifacts[name] = sum(p.stat().st_size for p in (output / name).iterdir())
    results, frames, comparisons = {}, {}, {}
    selected = None
    for split, target, history in [('validation', validation, train), ('test', test, pd.concat([train, validation]))]:
        results[split], frames[split], comparisons[split] = {}, {}, {}
        for name, model in models.items():
            print(f'Evaluating {split}/{name}', flush=True)
            summary, per_user = evaluate(model, history, target, movies, config['k'], config['seed'], config['bootstrap_samples'])
            per_user['training_positive_count'] = per_user.user_id.map(training_counts).fillna(0).astype(int)
            per_user['fallback_eligible'] = per_user.training_positive_count < threshold
            summary['cohorts'] = {}
            for cohort, mask in [('warm', ~per_user.fallback_eligible), ('fallback_eligible', per_user.fallback_eligible)]:
                subset = per_user[mask]
                summary['cohorts'][cohort] = {'users': len(subset), 'metrics':
                    {metric: float(subset[metric].mean()) if len(subset) else None for metric in METRICS}}
            per_user.to_csv(output / f'{split}_{name}_per_user.csv', index=False)
            results[split][name], frames[split][name] = summary, per_user
        for control, treatment in [('popularity', 'item_cf'), ('popularity', 'implicit_als'), ('item_cf', 'implicit_als')]:
            comparisons[split][f'{treatment}_vs_{control}'] = paired_comparison(
                frames[split][control], frames[split][treatment], config['seed'], config['bootstrap_samples'])
        if split == 'validation':
            selected = max(results[split], key=lambda name: results[split][name]['metrics']['ndcg']['mean'])
            write(output / 'selection.json', {'metric': 'validation macro NDCG@K', 'selected_model': selected,
                  'note': 'One preconfigured candidate per family; no test-driven hyperparameter tuning.'})
    write(output / 'metrics.json', results)
    write(output / 'paired_comparisons.json', comparisons)
    write(output / 'manifest.json', {'schema_version': 1, 'config': config, 'cutoffs': cutoffs,
          'rows': {'train': len(train), 'validation': len(validation), 'test': len(test)},
          'fit_seconds': fit_times, 'artifact_bytes': artifacts, 'selected_on_validation': selected,
          'total_seconds': time.perf_counter() - started,
          'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__},
          'thread_environment': {key: os.environ.get(key) for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']},
          'source_sha256': {p.name: digest(p) for p in sorted(directory.iterdir()) if p.is_file()},
          'code_sha256': {p.name: digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
          'artifact_sha256': {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob('*')) if p.is_file()},
          'protocol': 'Frozen train-only profiles and weights; full train catalog; validation/test seen filters identical across models.'})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/phase2.json'))
    parser.add_argument('--data', type=Path, default=Path('data/raw/ml-1m'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run_benchmark(json.loads(args.config.read_text()), args.data, args.output)
    print(f'Benchmark saved to {args.output}')


if __name__ == '__main__':
    main()
