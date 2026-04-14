from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from base import Base
from dotenv import load_dotenv
import os

load_dotenv()

DB_URL = os.getenv("DB_URL")

engine = create_async_engine(
    DB_URL,
    echo=False,
    connect_args={"ssl": True, "prepared_statement_cache_size": 0},
)

AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session