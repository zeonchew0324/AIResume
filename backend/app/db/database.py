from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
import os

DB_URL = os.get_env("DB_URL")

engine = create_async_engine(DB_URL, echo=False)

AsyncSessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

class Base(declarative_base):
    pass

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session