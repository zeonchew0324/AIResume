import pytest
from unittest.mock import AsyncMock, Mock, patch

from app.services.resume_service import (
    MAX_RESUME_BYTES,
    MAX_RESUME_TEXT_CHARS,
    upload_resumes,
)


def fake_upload(content: bytes):
    upload = Mock()
    upload.read = AsyncMock(side_effect=lambda n=-1: content if n < 0 else content[:n])
    return upload

def fake_db():
    return Mock(add=Mock(), commit=AsyncMock(), refresh=AsyncMock())


async def test_rejects_file_over_size_cap():
    upload = fake_upload(b"x" * (MAX_RESUME_BYTES + 1))
    with pytest.raises(ValueError, match="too large"):
        await upload_resumes(fake_db(), "My Resume", upload, "user-1")

@patch("app.services.resume_service.extract_text_from_pdf", return_value="a" * (MAX_RESUME_TEXT_CHARS + 1))
async def test_rejects_extracted_text_over_cap(mock_extract):
    upload = fake_upload(b"%PDF-1.4 small file")
    with pytest.raises(ValueError, match="too long"):
        await upload_resumes(fake_db(), "My Resume", upload, "user-1")

@patch("app.services.resume_service.extract_text_from_pdf", side_effect=Exception("broken xref table"))
async def test_unparseable_pdf_raises_clean_error(mock_extract):
    upload = fake_upload(b"not a pdf at all")
    with pytest.raises(ValueError, match="Failed to extract text"):
        await upload_resumes(fake_db(), "My Resume", upload, "user-1")

@patch("app.services.resume_service.extract_text_from_pdf", return_value="Sample resume text")
async def test_upload_within_limits_persists_row(mock_extract):
    db = fake_db()
    row = await upload_resumes(db, "  My Resume  ", fake_upload(b"%PDF-1.4 ok"), "user-1")
    assert row.name == "My Resume"
    assert row.resume_text == "Sample resume text"
    assert row.user_id == "user-1"
    db.add.assert_called_once_with(row)
    db.commit.assert_awaited_once()
