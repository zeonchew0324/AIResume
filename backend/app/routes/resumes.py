from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from app.services.resume_service import get_saved_resumes, delete_resume_service, upload_resumes
from app.db.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession

import logging
import traceback

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/api/resumes")
async def upload_resume(
    db: AsyncSession = Depends(get_db),
    name: str = Form(...),
    resume: UploadFile = File(...),
    user_id: str = Form(...),
):
    try:
        row = await upload_resumes(db, name, resume, user_id)
        return {"message": "Resume uploaded successfully", "resume_id": str(row.id)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error uploading resume: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Failed to upload resume. Please try again.")


@router.get("/api/resumes")
async def list_resumes(user_id: str, db: AsyncSession = Depends(get_db)):
    try:
        resumes = await get_saved_resumes(db, user_id)
        return {"resumes": [{
            "id": str(r.id),
            "name": r.name,
            "created_at": r.created_at.isoformat()
        } for r in resumes]}
    except Exception as e:
        logger.error(f"Error fetching resumes: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Failed to fetch resumes. Please try again.")


@router.delete("/api/resumes/{resume_id}", status_code=204)
async def delete_resume(resume_id: str, user_id: str, db: AsyncSession = Depends(get_db)):
    try:
        if await delete_resume_service(db, resume_id, user_id):
            return None
        raise HTTPException(status_code=404, detail="Resume not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting resume: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Failed to delete resume. Please try again.")
