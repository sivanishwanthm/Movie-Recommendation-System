"""Descriptive rating and item drift; never an automatic retraining trigger."""
import argparse
import json
import os
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from ..data import digest


def validate(frame):
    if frame.empty or not {'rating', 'movie_id'}.issubset(frame):
        raise ValueError('nonempty rating and movie_id columns required')
    ratings = frame.rating.to_numpy(dtype=float)
    items = frame.movie_id.to_numpy(dtype=float)
    if (not np.isfinite(ratings).all() or not np.isfinite(items).all()
            or not ((ratings >= .5) & (ratings <= 5) & (ratings * 2 % 1 == 0)).all()
            or not ((items > 0) & (items % 1 == 0)).all()):
        raise ValueError('invalid ratings or item IDs')


def compare(reference, current, minimum=100):
    validate(reference)
    validate(current)
    if minimum < 1:
        raise ValueError('minimum must be positive')
    bins = np.arange(.5, 5.5, .5)
    a = reference.rating.value_counts().reindex(bins, fill_value=0).to_numpy()
    b = current.rating.value_counts().reindex(bins, fill_value=0).to_numpy()
    # Aggregate over a fixed reference head plus tail and unseen buckets.
    head = reference.movie_id.value_counts().sort_index().sort_values(ascending=False, kind='stable').head(100).index
    known = set(reference.movie_id)
    def item_counts(frame):
        counts = frame.movie_id.value_counts()
        return np.r_[counts.reindex(head, fill_value=0).to_numpy(),
                     ((~frame.movie_id.isin(head)) & frame.movie_id.isin(known)).sum(),
                     (~frame.movie_id.isin(known)).sum()]
    rating_js = float(jensenshannon(a, b, base=2) ** 2)
    item_js = float(jensenshannon(item_counts(reference), item_counts(current), base=2) ** 2)
    enough = min(len(reference), len(current)) >= minimum
    return {'reference_rows': len(reference), 'current_rows': len(current), 'minimum_rows': minimum,
            'status': ('alert' if max(rating_js, item_js) > .1 else 'ok') if enough else 'insufficient_data',
            'rating_js_divergence_bits': rating_js, 'item_js_divergence_bits': item_js,
            'unseen_item_fraction': float((~current.movie_id.isin(known)).mean()),
            'threshold_bits': .1,
            'interpretation': 'Heuristic distribution alert, not a significance test or recommendation-quality estimate. '
                              'MovieLens ratings and live profile ratings have different selection mechanisms. '
                              'Review cohort, volume and instrumentation before deciding to retrain.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True, help='Training ratings CSV')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--current', type=Path, help='Comparable ratings CSV')
    source.add_argument('--live-since', help='Read live rating events since an ISO timestamp with timezone')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--minimum', type=int, default=100)
    args = parser.parse_args()
    reference = pd.read_csv(args.reference)
    validate(reference)
    if args.minimum < 1:
        raise ValueError('minimum must be positive')
    metadata = {'reference_sha256': digest(args.reference), 'generated_at': datetime.now(timezone.utc).isoformat()}
    if args.current:
        current = pd.read_csv(args.current)
        metadata.update(source='offline_csv', current_sha256=digest(args.current))
    else:
        from sqlalchemy import select
        from ..serving.database import connect, Event
        since = datetime.fromisoformat(args.live_since.replace('Z', '+00:00'))
        if since.tzinfo is None:
            raise ValueError('live-since must include timezone')
        until = datetime.now(timezone.utc)
        if since >= until:
            raise ValueError('live-since must be before now')
        engine = connect(os.environ['DATABASE_URL'])
        try:
            with engine.connect() as conn:
                current = pd.read_sql(select(Event.movie_id, Event.value.label('rating')).where(
                    Event.kind == 'rating', Event.created_at >= since, Event.created_at < until), conn)
        finally:
            engine.dispose()
        metadata.update(source='live_rating_events', since=since.isoformat(), until=until.isoformat())
    result = compare(reference, current, args.minimum) if not current.empty else {
        'status':'insufficient_data', 'reference_rows':len(reference), 'current_rows':0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as out:
        json.dump(dict(metadata, **result), out, indent=2)
        out.write('\n')


if __name__ == '__main__':
    main()
