from types import SimpleNamespace
from unittest.mock import patch

from fastapi import Request

from app.auth import get_current_user_id
from app.limiter import limiter, user_or_ip
from app.main import app
from app.models.schemas import ResumeAnalysisResponse


def make_request(user_id=None, ip="203.0.113.7"):
    state = SimpleNamespace()
    if user_id is not None:
        state.user_id = user_id
    return SimpleNamespace(state=state, client=SimpleNamespace(host=ip))

def test_authenticated_requests_keyed_by_user():
    assert user_or_ip(make_request(user_id="user-123")) == "user-123"

def test_same_user_from_different_ips_shares_one_bucket():
    a = user_or_ip(make_request(user_id="user-123", ip="203.0.113.7"))
    b = user_or_ip(make_request(user_id="user-123", ip="198.51.100.9"))
    assert a == b

def test_anonymous_requests_fall_back_to_ip():
    assert user_or_ip(make_request()) == "203.0.113.7"


MOCK_ANALYSIS = ResumeAnalysisResponse(
    match_score=50.0, feedback="ok", missing_keywords=[], suggestions=[], score_breakdown=[]
)

def override_user(user_id: str):
    """Auth override that also stamps request.state like the real dependency."""
    async def _override(request: Request) -> str:
        request.state.user_id = user_id
        return user_id
    return _override

# End-to-end check of the ordering this feature relies on: the auth dependency
# must populate request.state.user_id before slowapi evaluates the key.
@patch("app.routes.ats.analyze_resume_service", return_value=MOCK_ANALYSIS)
async def test_sixth_request_within_a_minute_is_limited_per_user(mock_service, client):
    data = {"resume_id": "r", "job_title": "t", "job_description": "d"}
    limiter.enabled = True
    try:
        app.dependency_overrides[get_current_user_id] = override_user("rate-user-a")
        codes = [(await client.post("/api/analyze", data=data)).status_code for _ in range(6)]
        assert codes == [200] * 5 + [429]

        # A different user is not affected by user A exhausting their bucket
        app.dependency_overrides[get_current_user_id] = override_user("rate-user-b")
        assert (await client.post("/api/analyze", data=data)).status_code == 200
    finally:
        limiter.enabled = False
