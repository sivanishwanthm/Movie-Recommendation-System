"""Compose bootstrap: download official data, benchmark, verify and select first release."""
import json
from pathlib import Path
from recsyslab.data import download
from recsyslab.benchmark import run_benchmark
from recsyslab.operations.releases import prepare, select_release, selected_path

root = Path('artifacts/releases')
if (root / 'current.json').exists():
    print(f'Existing verified release retained: {selected_path(root)}')
else:
    data = download('1m', Path('data/raw'))
    run = Path('artifacts/compose-baseline')
    if not run.exists():
        run_benchmark(json.loads(Path('configs/phase2.json').read_text()), data, run)
    receipt = prepare(run, root)
    print(json.dumps(select_release(root, receipt['release']), indent=2))
