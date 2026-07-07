import asyncio

from unittest.mock import patch
from app.main import app
from app.auth import get_current_user_id
from app.models.schemas import ResumeImprovementResponse


MOCK_IMPROVEMENT = ResumeImprovementResponse(
    improved_resume="John Doe\nBackend Engineer with FastAPI and Kubernetes experience...",
    changes=[
        {"section": "Skills", "change": "Added Kubernetes and gRPC to the skills list."},
        {"section": "Experience", "change": "Quantified the impact of the API migration project."},
    ],
    keywords_added=["Kubernetes", "gRPC"],
)

def make_data(**overrides):
    data = {
        "resume_id": "00000000-0000-0000-0000-000000000001",
        "job_title": "Backend Engineer",
        "job_description": "We need a Python developer with FastAPI and Kubernetes experience.",
        "extra_info": "I led the migration to microservices.",
    }
    data.update(overrides)
    return data

# Test 1: Happy path
@patch("app.routes.ats.improve_resume_service", return_value=MOCK_IMPROVEMENT)
async def test_improve_success(mock_service, client):
    response = await client.post("/api/improve", data=make_data())
    data = response.json()
    assert response.status_code == 200
    assert data["improved_resume"].startswith("John Doe")
    assert len(data["changes"]) == 2
    assert data["keywords_added"] == ["Kubernetes", "gRPC"]

# Test 2: extra_info is optional and defaults to empty
@patch("app.routes.ats.improve_resume_service", return_value=MOCK_IMPROVEMENT)
async def test_improve_without_extra_info(mock_service, client):
    data = make_data()
    del data["extra_info"]
    response = await client.post("/api/improve", data=data)
    assert response.status_code == 200
    assert mock_service.call_args.args[3] == ""

# Test 3: Empty/unreadable resume → 400
@patch("app.routes.ats.improve_resume_service", side_effect=ValueError("Resume is empty or unreadable"))
async def test_improve_unreadable_resume(mock_service, client):
    response = await client.post("/api/improve", data=make_data())
    assert response.status_code == 400
    assert response.json()["detail"] == "Resume is empty or unreadable"

# Test 4: Internal error → 500, raw error not exposed
@patch("app.routes.ats.improve_resume_service", side_effect=Exception("OpenAI rate limit exceeded"))
async def test_improve_server_error(mock_service, client):
    response = await client.post("/api/improve", data=make_data())
    assert response.status_code == 500
    assert response.json()["detail"] == "Improvement failed. Please try again."
    assert "OpenAI" not in response.json()["detail"]

# Test 5: AI model timeout → 504
@patch("app.routes.ats.improve_resume_service", side_effect=asyncio.TimeoutError())
async def test_improve_timeout(mock_service, client):
    response = await client.post("/api/improve", data=make_data())
    assert response.status_code == 504
    assert response.json()["detail"] == "The AI model took too long. Please try again."

# Test 6: Missing required field → 422
async def test_improve_missing_field(client):
    response = await client.post("/api/improve",
        data={"job_description": "Some job description"},
    )
    assert response.status_code == 422

# Test 7: Missing auth token → 401
async def test_improve_requires_auth(client):
    app.dependency_overrides.pop(get_current_user_id, None)
    response = await client.post("/api/improve", data=make_data())
    assert response.status_code == 401
