import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch
from app.main import app
from app.auth import get_current_user_id

# Disable rate limiting for all tests
@pytest.fixture(autouse=True)
def disable_rate_limit():
    with patch("app.routes.ats.limiter.limit", return_value=lambda f: f):
        yield

# Treat every request as authenticated unless a test removes the override
@pytest.fixture(autouse=True)
def override_auth():
    app.dependency_overrides[get_current_user_id] = lambda: "test-user-id"
    yield
    app.dependency_overrides.pop(get_current_user_id, None)

@pytest_asyncio.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
