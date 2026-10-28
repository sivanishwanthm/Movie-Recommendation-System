"""Descriptive EDA only; held-out summaries never feed model features."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def summarize(ratings: pd.DataFrame, movies: pd.DataFrame, output: Path) -> dict:
    per_user = ratings.groupby('user_id').size()
    per_movie = ratings.groupby('movie_id').size()
    dates = pd.to_datetime(ratings.timestamp, unit='s', utc=True)
    monthly = (dates.dt.year * 100 + dates.dt.month).value_counts().sort_index()
    monthly.index = [f'{value // 100:04d}-{value % 100:02d}' for value in monthly.index]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ratings.rating.value_counts().sort_index().plot.bar(ax=axes[0, 0], title='Observed rating distribution')
    per_user.plot.hist(bins=50, ax=axes[0, 1], title='Ratings per user')
    per_movie.plot.hist(bins=50, ax=axes[1, 0], title='Ratings per rated movie')
    monthly.plot(ax=axes[1, 1], title='Observed ratings by UTC month', rot=45)
    fig.tight_layout()
    fig.savefig(output / 'eda.png', dpi=150)
    plt.close(fig)
    return {'ratings': len(ratings), 'users': int(ratings.user_id.nunique()),
            'movies_in_metadata': len(movies), 'rated_movies': int(ratings.movie_id.nunique()),
            'sparsity': 1 - len(ratings) / (ratings.user_id.nunique() * len(movies)),
            'rating_mean': float(ratings.rating.mean()), 'rating_std': float(ratings.rating.std()),
            'rating_counts': {str(k): int(v) for k, v in ratings.rating.value_counts().sort_index().items()},
            'ratings_per_user': per_user.describe().to_dict(), 'ratings_per_movie': per_movie.describe().to_dict(),
            'timestamp_min_utc': str(pd.to_datetime(ratings.timestamp.min(), unit='s', utc=True)),
            'timestamp_max_utc': str(pd.to_datetime(ratings.timestamp.max(), unit='s', utc=True)),
            'interpretation': 'Observational explicit ratings; missing ratings are not negative outcomes.'}
