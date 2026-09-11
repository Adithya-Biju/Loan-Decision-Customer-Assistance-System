from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.database import create_tables


@asynccontextmanager
async def lifespan(app: FastAPI):

    create_tables()

    yield