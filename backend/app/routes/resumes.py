from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from app.services.resume_service import get_saved_resumes, delete_resume_service, upload_resumes
from app.db.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession

import logging
import traceback

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/api/resumes")
async def upload_resume(db: AsyncSession = Depends(get_db), name: str = Form(...), resume: UploadFile = File(...), user_id: str = Form(...)):
    try:
        resume = await upload_resumes(db, name, resume, user_id)
        return {"message": "Resume uploaded successfully", "resume_id": str(resume.id)}
    except Exception as e:
        logger.error(f"Error occurred while uploading resume: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Failed to upload resume. Please try again.")


@router.get("/api/resumes")
async def list_resumes(db: AsyncSession = Depends(get_db)):
    try:
        resumes =  await get_saved_resumes(db)
        return {"resumes": [{
            "id": str(r.id),
            "name": r.name,
            "created_at": r.created_at.isoformat()
        } for r in resumes ]}
    except Exception as e:
        logger.error(f"Error occurred while fetching resumes: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Failed to fetch resumes. Please try again.")

@router.delete("/api/resumes/{resume_id}", status_code=204)
async def delete_resume(resume_id: str, db: AsyncSession = Depends(get_db)):
    try:
        if await delete_resume_service(db, resume_id):
            return None
        else:
            raise HTTPException(status_code=404, detail="Resume cannot be deleted")
    except HTTPException as e:
        raise 
    except Exception as e:
        logger.error(f"Error occurred while deleting resume: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Failed to delete resume. Please try again.")