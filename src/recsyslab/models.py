"""Train-only popularity baseline, also the future cold-start fallback."""
from dataclasses import dataclass
from typing import Set, List, Optional, Protocol
import pandas as pd


@dataclass
class Popularity:
    catalog: List[int]
    counts: dict
    threshold: float

    @classmethod
    def fit(cls, train: pd.DataFrame, threshold: float = 4.0):
        counts = train.loc[train.rating >= threshold].groupby('movie_id').size().to_dict()
        catalog = sorted(map(int, train.movie_id.unique()), key=lambda i: (-counts.get(i, 0), i))
        return cls(catalog, {int(k): int(v) for k, v in counts.items()}, threshold)

    def recommend(self, seen: Set[int], k: int, user_id: Optional[int] = None) -> List[int]:
        if k <= 0:
            raise ValueError('k must be positive')
        result = []
        for item in self.catalog:
            if item not in seen:
                result.append(item)
                if len(result) == k:
                    break
        return result


class Recommender(Protocol):
    catalog: List[int]
    counts: dict
    threshold: float

    def recommend(self, seen: Set[int], k: int, user_id: Optional[int] = None) -> List[int]: ...
