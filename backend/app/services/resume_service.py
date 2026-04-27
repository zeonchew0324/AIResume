from sqlalchemy.ext.asyncio import AsyncSession
from app.models.db import Resume
from sqlalchemy import select, delete
from app.utils.pdf_parser import extract_text_from_pdf
from fastapi import UploadFile
import io

async def get_saved_resumes(db: AsyncSession) -> list[Resume]:
    # Placeholder for future implementation of fetching saved resumes
    result = await db.execute(select(Resume).order_by(Resume.created_at.desc()))

    return result.scalars().all()

async def upload_resumes(db: AsyncSession, name: str, resume: UploadFile, user_id: str | None = None) -> Resume:
    try:
        file_bytes = await resume.read()
        resume_text = extract_text_from_pdf(io.BytesIO(file_bytes))
    except Exception as e:
        raise ValueError("Failed to extract text from resume. Please ensure the file is a valid PDF.") from e
    
    row = Resume(name=name, resume_text=resume_text, user_id=user_id)
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row

async def delete_resume_service(db: AsyncSession, resume_id: str):
    result = await db.execute(delete(Resume).where(Resume.id == resume_id))
    await db.commit()
    return result.rowcount > 0  # Return True if a row was deleted, False otherwise