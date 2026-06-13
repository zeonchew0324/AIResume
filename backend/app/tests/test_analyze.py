import pytest
from unittest.mock import patch
from app.main import app
from app.auth import get_current_user_id
from app.models.schemas import ResumeAnalysisResponse


MOCK_ANALYSIS = ResumeAnalysisResponse(
    match_score=78.0,
    feedback="Strong Python background but missing cloud experience.",
    missing_keywords=["Kubernetes", "gRPC"],
    suggestions=[{"focus_area": "Skills", "advice": "Add Kubernetes to your skills section."}],
    score_breakdown=[
        {"category": "Technical Skills", "score": 80, "reason": "Strong Python and FastAPI but Docker not mentioned."},
        {"category": "Experience", "score": 75, "reason": "4 years aligns but role expects 5."},
        {"category": "Keywords", "score": 70, "reason": "Missing distributed systems and gRPC."},
        {"category": "Soft Skills", "score": 60, "reason": "No mention of cross-team collaboration."},
    ]
)

def make_data():
    return {
        "resume_id": "00000000-0000-0000-0000-000000000001",
        "job_title": "Backend Engineer",
        "job_description": "We need a Python developer with FastAPI and Kubernetes experience.",
    }

# Test 1: Happy path
@patch("app.routes.ats.analyze_resume_service", return_value=MOCK_ANALYSIS)
async def test_analyze_success(mock_service, client):
    response = await client.post("/api/analyze", data=make_data())
    data = response.json()
    assert response.status_code == 200
    assert data["match_score"] == 78.0
    assert data["missing_keywords"] == ["Kubernetes", "gRPC"]
    assert len(data["score_breakdown"]) == 4

# Test 2: Empty/unreadable resume → 400
@patch("app.routes.ats.analyze_resume_service", side_effect=ValueError("Resume is empty or unreadable"))
async def test_analyze_unreadable_resume(mock_service, client):
    response = await client.post("/api/analyze", data=make_data())
    assert response.status_code == 400
    assert response.json()["detail"] == "Resume is empty or unreadable"

# Test 3: Internal error → 500, raw error not exposed
@patch("app.routes.ats.analyze_resume_service", side_effect=Exception("OpenAI rate limit exceeded"))
async def test_analyze_server_error(mock_service, client):
    response = await client.post("/api/analyze", data=make_data())
    assert response.status_code == 500
    assert response.json()["detail"] == "Analysis failed. Please try again."
    assert "OpenAI" not in response.json()["detail"]

# Test 4: Missing required field → 422
async def test_analyze_missing_field(client):
    response = await client.post("/api/analyze",
        data={"job_description": "Some job description"},
    )
    assert response.status_code == 422

# Test 5: Missing auth token → 401
async def test_analyze_requires_auth(client):
    app.dependency_overrides.pop(get_current_user_id, None)
    response = await client.post("/api/analyze", data=make_data())
    assert response.status_code == 401
