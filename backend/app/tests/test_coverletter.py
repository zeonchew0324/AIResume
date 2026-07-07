import asyncio

from unittest.mock import patch
from app.main import app
from app.auth import get_current_user_id


MOCK_COVER_LETTER = "Dear Hiring Manager,\n\nI am excited to apply to Acme Corp..."

def make_data(**overrides):
    data = {
        "resume_id": "00000000-0000-0000-0000-000000000001",
        "job_title": "Backend Engineer",
        "job_description": "We need a Python developer with FastAPI and Kubernetes experience.",
        "company_name": "Acme Corp",
        "extra_info": "I admire the company's open-source work.",
    }
    data.update(overrides)
    return data

# Test 1: Happy path
@patch("app.routes.ats.generate_coverletter", return_value=MOCK_COVER_LETTER)
async def test_coverletter_success(mock_service, client):
    response = await client.post("/api/coverletter", data=make_data())
    assert response.status_code == 200
    assert response.json() == {"cover_letter": MOCK_COVER_LETTER}

# Test 2: extra_info is optional and defaults to empty
@patch("app.routes.ats.generate_coverletter", return_value=MOCK_COVER_LETTER)
async def test_coverletter_without_extra_info(mock_service, client):
    data = make_data()
    del data["extra_info"]
    response = await client.post("/api/coverletter", data=data)
    assert response.status_code == 200
    assert mock_service.call_args.args[4] == ""

# Test 3: Empty/unreadable resume → 400
@patch("app.routes.ats.generate_coverletter", side_effect=ValueError("Resume is empty or unreadable"))
async def test_coverletter_unreadable_resume(mock_service, client):
    response = await client.post("/api/coverletter", data=make_data())
    assert response.status_code == 400
    assert response.json()["detail"] == "Resume is empty or unreadable"

# Test 4: Internal error → 500, raw error not exposed
@patch("app.routes.ats.generate_coverletter", side_effect=Exception("OpenAI rate limit exceeded"))
async def test_coverletter_server_error(mock_service, client):
    response = await client.post("/api/coverletter", data=make_data())
    assert response.status_code == 500
    assert response.json()["detail"] == "Cover letter generation failed. Please try again."
    assert "OpenAI" not in response.json()["detail"]

# Test 5: AI model timeout → 504
@patch("app.routes.ats.generate_coverletter", side_effect=asyncio.TimeoutError())
async def test_coverletter_timeout(mock_service, client):
    response = await client.post("/api/coverletter", data=make_data())
    assert response.status_code == 504
    assert response.json()["detail"] == "The AI model took too long. Please try again."

# Test 6: Missing company_name → 422
async def test_coverletter_missing_company(client):
    data = make_data()
    del data["company_name"]
    response = await client.post("/api/coverletter", data=data)
    assert response.status_code == 422

# Test 7: Missing auth token → 401
async def test_coverletter_requires_auth(client):
    app.dependency_overrides.pop(get_current_user_id, None)
    response = await client.post("/api/coverletter", data=make_data())
    assert response.status_code == 401
