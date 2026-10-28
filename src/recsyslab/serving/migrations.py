"""Explicit transactional v1 -> v2 additive PostgreSQL migration."""
import os
from sqlalchemy import inspect, text
from .database import Base, connect


def upgrade(engine):
    with engine.begin() as connection:
        connection.execute(text('SELECT pg_advisory_xact_lock(52520302)'))
        current=[]
        if inspect(connection).has_table('schema_version'):
            current=connection.execute(text('SELECT version FROM schema_version')).scalars().all()
        if current not in ([],[1],[2]):
            raise ValueError('unsupported schema version; migration refused')
        Base.metadata.create_all(connection)
        if not current:
            connection.execute(text('INSERT INTO schema_version (version) VALUES (2)'))
        elif current==[1]:
            connection.execute(text('UPDATE schema_version SET version=2 WHERE version=1'))
    return 2


def main():
    engine=connect(os.environ['DATABASE_URL'])
    print(f'Database schema ready: v{upgrade(engine)} (existing live records preserved)')
    engine.dispose()

if __name__=='__main__':main()
