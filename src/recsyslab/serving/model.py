"""Immutable item-CF artifact; fold in live ratings without training."""
import json
import hashlib
from pathlib import Path
import numpy as np
from .database import Rating
from ..classical import load_model, ItemCF
from ..data import digest


class LiveRecommender:
    def __init__(self, run_directory: Path):
        manifest=json.loads((run_directory/'manifest.json').read_text())
        expected=manifest['artifact_sha256']
        names=['model.json','arrays.npz','profiles.npz','similarity.npz']
        for name in names:
            key=f'item_cf/{name}'
            if key not in expected or digest(run_directory/key)!=expected[key]:
                raise ValueError(f'item-CF artifact integrity check failed: {name}')
        self.model=load_model(run_directory/'item_cf')
        if not isinstance(self.model,ItemCF): raise ValueError('expected item-CF artifact')
        self.version='item-cf-'+hashlib.sha256(json.dumps({name:expected['item_cf/'+name] for name in names},sort_keys=True).encode()).hexdigest()[:16]
        self.item_index={int(item):i for i,item in enumerate(self.model.items)}
        self.available=set(self.item_index)
        self.catalog_sha256=digest(run_directory/'item_cf/arrays.npz')

    def recommend(self, ratings: list, k: int):
        seen={int(r.movie_id) for r in ratings}
        positives=[self.item_index[r.movie_id] for r in ratings if r.value>=self.model.threshold and r.movie_id in self.item_index]
        if len(positives)<self.model.min_positives:
            return self.model.baseline.recommend(seen,k), 'popularity_fallback',len(positives)
        scores=np.asarray(self.model.similarity[positives].sum(axis=0)).ravel()
        order=np.lexsort((self.model.items,-self.model.popularity,-scores))
        result=[int(self.model.items[i]) for i in order if int(self.model.items[i]) not in seen][:k]
        return result,'item_cf_live_profile',len(positives)
