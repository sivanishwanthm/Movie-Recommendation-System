"""Record the currently selected local release in MLflow; does not activate it."""
import json
import os
from pathlib import Path
from recsyslab.operations.releases import track

root = Path(os.getenv('RELEASE_ROOT', 'artifacts/releases'))
release = json.loads((root / 'current.json').read_text())['release']
print(json.dumps(track(root, release, os.getenv('MLFLOW_TRACKING_URI', 'sqlite:///artifacts/mlflow.db')), indent=2))
