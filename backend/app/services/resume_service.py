from sqlalchemy.ext.asyncio import AsyncSession
from app.models.db import Resume
from sqlalchemy import select, delete
from app.utils.pdf_parser import extract_text_from_pdf
from app.utils.input_cleaner import clean_input
from fastapi import UploadFile
import io

# Uploads are read fully into memory before parsing, so cap the file size;
# cap the extracted text too since it goes verbatim into LLM prompts.
MAX_RESUME_BYTES = 5 * 1024 * 1024  # 5 MB
MAX_RESUME_TEXT_CHARS = 50_000

async def get_saved_resumes(db: AsyncSession, user_id: str) -> list[Resume]:
    result = await db.execute(
        select(Resume).where(Resume.user_id == user_id).order_by(Resume.created_at.desc())
    )
    return result.scalars().all()

async def get_resume_text(db: AsyncSession, resume_id: str, user_id: str) -> str:
    result = await db.execute(
        select(Resume).where(Resume.id == resume_id, Resume.user_id == user_id)
    )
    resume = result.scalar_one_or_none()
    if resume is None:
        raise ValueError("Resume not found")
    return resume.resume_text

async def upload_resumes(db: AsyncSession, name: str, resume: UploadFile, user_id: str) -> Resume:
    name = clean_input(name, 200)

    # Read one byte past the cap so we can tell "at the limit" from "over it"
    # without pulling an arbitrarily large body into memory.
    file_bytes = await resume.read(MAX_RESUME_BYTES + 1)
    if len(file_bytes) > MAX_RESUME_BYTES:
        raise ValueError("Resume file is too large (max 5 MB).")

    try:
        resume_text = extract_text_from_pdf(io.BytesIO(file_bytes))
    except Exception as e:
        raise ValueError("Failed to extract text from resume.") from e

    if len(resume_text) > MAX_RESUME_TEXT_CHARS:
        raise ValueError("Resume is too long. Please upload a shorter document.")

    row = Resume(name=name, resume_text=resume_text, user_id=user_id)
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row

async def delete_resume_service(db: AsyncSession, resume_id: str, user_id: str) -> bool:
    result = await db.execute(
        delete(Resume).where(Resume.id == resume_id, Resume.user_id == user_id)
    )
    await db.commit()
    return result.rowcount > 0
