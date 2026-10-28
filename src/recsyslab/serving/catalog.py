"""Explicit schema initialization and metadata-only MovieLens import."""
import argparse
import os
from pathlib import Path
import pandas as pd
from sqlalchemy.dialects.postgresql import insert
from .database import connect, initialize, Movie


def import_movies(engine, path: Path) -> int:
    if path.suffix == '.dat':
        movies = pd.read_csv(path, sep='::', engine='python', encoding='latin-1', names=['id', 'title', 'genres'])
    else:
        movies = pd.read_csv(path).rename(columns={'movieId': 'id'})
    if movies[['id','title','genres']].isna().any().any() or movies.id.duplicated().any():
        raise ValueError('invalid movie catalog')
    rows = [{'id': int(r.id), 'title': r.title, 'genres': r.genres.split('|')} for r in movies.itertuples()]
    with engine.begin() as connection:
        for start in range(0,len(rows),500):
            statement = insert(Movie).values(rows[start:start+500])
            connection.execute(statement.on_conflict_do_update(index_elements=['id'],set_={'title':statement.excluded.title,'genres':statement.excluded.genres}))
    return len(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--movies',type=Path,default=Path('data/raw/ml-1m/movies.dat'))
    args=parser.parse_args()
    engine=connect(os.environ['DATABASE_URL'])
    initialize(engine)
    print(f'Imported {import_movies(engine,args.movies)} movie metadata rows; no historical ratings or users imported.')
    engine.dispose()

if __name__=='__main__':main()
