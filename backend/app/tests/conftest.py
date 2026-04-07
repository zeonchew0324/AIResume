import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch
from app.main import app

# Disable rate limiting for all tests
@pytest.fixture(autouse=True)
def disable_rate_limit():
    with patch("app.routes.ats.limiter.limit", return_value=lambda f: f):
        yield

@pytest_asyncio.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
