from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.db.base import Base
from dotenv import load_dotenv
import ssl
import os

load_dotenv()

DB_URL = os.getenv("DB_URL", "").replace("?sslmode=require", "")

ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE

engine = create_async_engine(
    DB_URL,
    echo=False,
    connect_args={
        "ssl": ssl_ctx,
        "prepared_statement_cache_size": 0,
        "server_settings": {"jit": "off"},
    },
)

AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session