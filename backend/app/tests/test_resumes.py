import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from app.main import app
from app.auth import get_current_user_id


def fake_resume(name="My Resume"):
    return SimpleNamespace(
        id=uuid.uuid4(),
        name=name,
        resume_text="Sample resume text",
        created_at=datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc),
    )

def make_upload():
    return {
        "data": {"name": "My Resume"},
        "files": {"resume": ("resume.pdf", b"%PDF-1.4 fake content", "application/pdf")},
    }

# --- POST /api/resumes ---

@patch("app.routes.resumes.upload_resumes")
async def test_upload_resume_success(mock_upload, client):
    row = fake_resume()
    mock_upload.return_value = row
    response = await client.post("/api/resumes", **make_upload())
    assert response.status_code == 200
    assert response.json() == {
        "message": "Resume uploaded successfully",
        "resume_id": str(row.id),
    }

@patch("app.routes.resumes.upload_resumes", side_effect=ValueError("Failed to extract text from resume."))
async def test_upload_resume_bad_pdf(mock_upload, client):
    response = await client.post("/api/resumes", **make_upload())
    assert response.status_code == 400
    assert response.json()["detail"] == "Failed to extract text from resume."

@patch("app.routes.resumes.upload_resumes", side_effect=Exception("db connection reset"))
async def test_upload_resume_server_error(mock_upload, client):
    response = await client.post("/api/resumes", **make_upload())
    assert response.status_code == 500
    assert "db connection" not in response.json()["detail"]

async def test_upload_resume_missing_file(client):
    response = await client.post("/api/resumes", data={"name": "My Resume"})
    assert response.status_code == 422

async def test_upload_resume_requires_auth(client):
    app.dependency_overrides.pop(get_current_user_id, None)
    response = await client.post("/api/resumes", **make_upload())
    assert response.status_code == 401

# --- GET /api/resumes ---

@patch("app.routes.resumes.get_saved_resumes")
async def test_list_resumes(mock_list, client):
    rows = [fake_resume("Resume A"), fake_resume("Resume B")]
    mock_list.return_value = rows
    response = await client.get("/api/resumes")
    assert response.status_code == 200
    resumes = response.json()["resumes"]
    assert [r["name"] for r in resumes] == ["Resume A", "Resume B"]
    assert resumes[0] == {
        "id": str(rows[0].id),
        "name": "Resume A",
        "created_at": "2026-07-01T12:00:00+00:00",
    }

@patch("app.routes.resumes.get_saved_resumes", return_value=[])
async def test_list_resumes_empty(mock_list, client):
    response = await client.get("/api/resumes")
    assert response.status_code == 200
    assert response.json() == {"resumes": []}

@patch("app.routes.resumes.get_saved_resumes", side_effect=Exception("db connection reset"))
async def test_list_resumes_server_error(mock_list, client):
    response = await client.get("/api/resumes")
    assert response.status_code == 500
    assert "db connection" not in response.json()["detail"]

async def test_list_resumes_requires_auth(client):
    app.dependency_overrides.pop(get_current_user_id, None)
    response = await client.get("/api/resumes")
    assert response.status_code == 401

# --- DELETE /api/resumes/{id} ---

@patch("app.routes.resumes.delete_resume_service", return_value=True)
async def test_delete_resume_success(mock_delete, client):
    response = await client.delete(f"/api/resumes/{uuid.uuid4()}")
    assert response.status_code == 204

@patch("app.routes.resumes.delete_resume_service", return_value=False)
async def test_delete_resume_not_found(mock_delete, client):
    response = await client.delete(f"/api/resumes/{uuid.uuid4()}")
    assert response.status_code == 404
    assert response.json()["detail"] == "Resume not found"

@patch("app.routes.resumes.delete_resume_service", side_effect=Exception("db connection reset"))
async def test_delete_resume_server_error(mock_delete, client):
    response = await client.delete(f"/api/resumes/{uuid.uuid4()}")
    assert response.status_code == 500
    assert "db connection" not in response.json()["detail"]

async def test_delete_resume_requires_auth(client):
    app.dependency_overrides.pop(get_current_user_id, None)
    response = await client.delete(f"/api/resumes/{uuid.uuid4()}")
    assert response.status_code == 401
