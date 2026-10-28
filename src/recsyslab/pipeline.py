"""Reproducible Phase 1 command line pipeline."""
import argparse
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
from .data import digest, download, load, temporal_split
from .eda import summarize
from .evaluation import evaluate
from .models import Popularity


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def run(config: dict, directory: Path, output: Path) -> dict:
    if not isinstance(config['k'], int) or config['k'] <= 0:
        raise ValueError('k must be a positive integer')
    if not isinstance(config['bootstrap_samples'], int) or config['bootstrap_samples'] < 1:
        raise ValueError('bootstrap_samples must be positive')
    if not 0.5 <= config['positive_threshold'] <= 5:
        raise ValueError('invalid positive threshold')
    output.mkdir(parents=True, exist_ok=False)  # Never silently overwrite a run.
    ratings, movies, users = load(directory, config['dataset'])
    train, val, test, cutoffs = temporal_split(ratings, config['train_fraction'], config['validation_fraction'])
    model = Popularity.fit(train, config['positive_threshold'])
    write(output / 'popularity.json', {'schema_version': 1, 'model': 'positive_count_popularity',
          'catalog': model.catalog, 'counts': model.counts, 'positive_threshold': model.threshold})
    write(output / 'eda.json', summarize(ratings, movies, output))
    results = {}
    for name, target, history in [('validation', val, train), ('test', test, pd.concat([train, val]))]:
        summary, per_user = evaluate(model, history, target, movies, config['k'], config['seed'], config['bootstrap_samples'])
        results[name] = summary
        per_user.to_csv(output / f'{name}_per_user.csv', index=False)
        target.to_csv(output / f'{name}_ratings.csv', index=False)
    train.to_csv(output / 'train_ratings.csv', index=False)
    write(output / 'metrics.json', results)
    manifest = {'schema_version': 1, 'config': config, 'cutoffs': cutoffs,
                'rows': {'train': len(train), 'validation': len(val), 'test': len(test)},
                'demographic_users': None if users is None else len(users),
                'source_sha256': {p.name: digest(p) for p in sorted(directory.iterdir()) if p.is_file()},
                'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__},
                'protocol': 'Global temporal split; frozen train model; test excludes train+validation history; full train catalog.',
                'artifact_sha256': digest(output / 'popularity.json')}
    write(output / 'manifest.json', manifest)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    fetch = sub.add_parser('download')
    fetch.add_argument('--dataset', choices=['1m', '32m'], default='1m')
    fetch.add_argument('--root', type=Path, default=Path('data/raw'))
    execute = sub.add_parser('run')
    execute.add_argument('--config', type=Path, default=Path('configs/phase1.json'))
    execute.add_argument('--data', type=Path, default=Path('data/raw/ml-1m'))
    execute.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'download':
        print(download(args.dataset, args.root))
    else:
        results = run(json.loads(args.config.read_text()), args.data, args.output)
        print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
