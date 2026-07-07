import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch
from app.main import app
from app.auth import get_current_user_id
from app.db.database import get_db

# Disable rate limiting for all tests. Patching limiter.limit does not work
# because the decorator is applied at import time; slowapi checks
# limiter.enabled on each request, so toggling that flag is the reliable way.
@pytest.fixture(autouse=True)
def disable_rate_limit():
    from app.limiter import limiter
    limiter.enabled = False
    yield
    limiter.enabled = True

# Treat every request as authenticated unless a test removes the override
@pytest.fixture(autouse=True)
def override_auth():
    app.dependency_overrides[get_current_user_id] = lambda: "test-user-id"
    yield
    app.dependency_overrides.pop(get_current_user_id, None)

# Avoid touching the real database: stub the session and the resume lookup
async def _fake_db():
    yield None

@pytest.fixture(autouse=True)
def override_db_and_resume_lookup():
    app.dependency_overrides[get_db] = _fake_db
    with patch("app.routes.ats.get_resume_text", return_value="Sample resume text"):
        yield
    app.dependency_overrides.pop(get_db, None)

@pytest_asyncio.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
