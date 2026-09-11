"""
SQLAlchemy engine and session factory.

This is the ONLY place a raw database connection is created.
Repositories receive a Session through the get_db dependency.
"""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings

settings = get_settings()

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    connect_args={"options": "-csearch_path=hdfc"},
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


class Base(DeclarativeBase):
    """Base class every ORM model inherits from."""

    pass


def create_tables() -> None:
    """Create all tables in the database."""
    from app.models import db_models

    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a request-scoped database session.
    """

    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()