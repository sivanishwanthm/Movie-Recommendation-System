"""PostgreSQL schema v2. Initialized explicitly, never during request handling."""
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, JSON, UniqueConstraint, CheckConstraint, create_engine, text
from sqlalchemy.orm import declarative_base

Base = declarative_base()

def now():
    return datetime.now(timezone.utc)


class SchemaVersion(Base):
    __tablename__ = 'schema_version'
    version = Column(Integer, primary_key=True)


class Profile(Base):
    __tablename__ = 'profiles'
    id = Column(String(36), primary_key=True)
    name = Column(String(60), nullable=False)
    token_hash = Column(String(64), unique=True, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), default=now, nullable=False)


class Movie(Base):
    __tablename__ = 'movies'
    id = Column(Integer, primary_key=True)
    title = Column(String(300), nullable=False)
    genres = Column(JSON, nullable=False)


class Rating(Base):
    __tablename__ = 'ratings'
    profile_id = Column(String(36), ForeignKey('profiles.id'), primary_key=True)
    movie_id = Column(Integer, ForeignKey('movies.id'), primary_key=True)
    value = Column(Float, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=now, nullable=False)
    __table_args__ = (CheckConstraint('value >= 0.5 AND value <= 5', name='rating_range'),)


class Batch(Base):
    __tablename__ = 'recommendation_batches'
    id = Column(String(36), primary_key=True)
    profile_id = Column(String(36), ForeignKey('profiles.id'), index=True, nullable=False)
    movie_ids = Column(JSON, nullable=False)
    model_version = Column(String(80), nullable=False)
    strategy = Column(String(60), nullable=False)
    latency_ms = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), default=now, nullable=False)


class Event(Base):
    __tablename__ = 'interaction_events'
    id = Column(String(36), primary_key=True)
    profile_id = Column(String(36), ForeignKey('profiles.id'), index=True, nullable=False)
    movie_id = Column(Integer, ForeignKey('movies.id'), nullable=False)
    batch_id = Column(String(36), ForeignKey('recommendation_batches.id'), nullable=True)
    kind = Column(String(20), nullable=False)
    value = Column(Float, nullable=True)
    dedupe_key = Column(String(120), nullable=True)
    created_at = Column(DateTime(timezone=True), default=now, nullable=False)
    __table_args__ = (UniqueConstraint('profile_id', 'dedupe_key', name='event_dedupe'),
                     CheckConstraint("kind IN ('impression', 'click', 'rating')", name='event_kind'))


def connect(url: str):
    if not url.startswith('postgresql+psycopg://'):
        raise ValueError('Phase 4 requires a PostgreSQL psycopg URL')
    return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=5)


def initialize(engine) -> None:
    from .migrations import upgrade
    upgrade(engine)


def check_schema(engine) -> None:
    with engine.connect() as connection:
        if connection.execute(text('SELECT version FROM schema_version')).scalars().all() != [2]:
            raise ValueError('upgrade database to schema v2 before serving')


class Experiment(Base):
    __tablename__ = 'experiments'
    key = Column(String(80), primary_key=True)
    specification = Column(JSON, nullable=False)
    fingerprint = Column(String(64), nullable=False)
    created_at = Column(DateTime(timezone=True), default=now, nullable=False)


class Assignment(Base):
    __tablename__ = 'experiment_assignments'
    id = Column(String(36), primary_key=True)
    experiment_key = Column(String(80), ForeignKey('experiments.key'), index=True, nullable=False)
    profile_id = Column(String(36), ForeignKey('profiles.id'), nullable=False)
    variant = Column(String(12), nullable=False)
    assigned_at = Column(DateTime(timezone=True), default=now, nullable=False)
    window_ends_at = Column(DateTime(timezone=True), nullable=False)
    __table_args__ = (UniqueConstraint('experiment_key','profile_id',name='one_variant_per_profile'),
                     CheckConstraint("variant IN ('control','treatment')",name='assignment_variant'))


class ExperimentBatch(Base):
    __tablename__ = 'experiment_batches'
    batch_id = Column(String(36), ForeignKey('recommendation_batches.id'), primary_key=True)
    assignment_id = Column(String(36), ForeignKey('experiment_assignments.id'), index=True, nullable=False)


class Exposure(Base):
    __tablename__ = 'experiment_exposures'
    assignment_id = Column(String(36), ForeignKey('experiment_assignments.id'), primary_key=True)
    batch_id = Column(String(36), ForeignKey('recommendation_batches.id'), nullable=False)
    exposed_at = Column(DateTime(timezone=True), default=now, nullable=False)


class ExperimentAttempt(Base):
    __tablename__ = 'experiment_attempts'
    id = Column(String(36), primary_key=True)
    assignment_id = Column(String(36), ForeignKey('experiment_assignments.id'), index=True, nullable=False)
    batch_id = Column(String(36), ForeignKey('recommendation_batches.id'), nullable=True)
    created_at = Column(DateTime(timezone=True), default=now, nullable=False)
    status_code = Column(Integer, nullable=True)
    api_latency_ms = Column(Float, nullable=True)
