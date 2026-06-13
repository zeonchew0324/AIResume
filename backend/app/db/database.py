from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.db.base import Base
from dotenv import load_dotenv
import ssl
import os
import certifi

load_dotenv()

DB_URL = os.getenv("DB_URL", "").replace("?sslmode=require", "")

# Verify the database server's TLS certificate (prevents man-in-the-middle on
# the DB connection). Defaults to the certifi CA bundle; set DB_SSL_ROOT_CERT to
# a specific root cert file (e.g. Supabase's downloaded CA) if your endpoint's
# certificate does not chain to a public root.
ssl_ctx = ssl.create_default_context(
    cafile=os.getenv("DB_SSL_ROOT_CERT") or certifi.where()
)

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