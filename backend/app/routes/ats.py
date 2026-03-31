from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.utils.pdf_parser import extract_text_from_pdf
from app.services.ats import analyze_resume_service 
from app.models.ats import ResumeAnalysisResponse

import logging 
import traceback

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/api/analyze")
async def analyze_resume(
    resume: UploadFile = File(...), 
    job_description: str = Form(...), 
    job_title: str = Form(...),
) -> ResumeAnalysisResponse:
    try:
        result = await analyze_resume_service(resume, job_description, job_title)
        return result
    except Exception as e:
        logger.error(f"Error occurred while analyzing resume: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))