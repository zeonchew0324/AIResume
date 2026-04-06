from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request 
from app.utils.pdf_parser import extract_text_from_pdf
from app.services.analyze_resume import analyze_resume_service
from app.services.improve_resume import improve_resume_service 
from app.models.ats import ResumeAnalysisResponse, ResumeImprovementResponse
from app.limiter import limiter


import logging 
import traceback

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/api/analyze")
@limiter.limit("5/minute")
async def analyze_resume(
    request: Request,
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

@router.post("/api/improve")
@limiter.limit("5/minute")
async def improve_resume(
    request: Request,
    resume: UploadFile = File(...),
    job_description: str = Form(...),
    job_title: str = Form(...),
    extra_info: str = Form("")
) -> ResumeImprovementResponse:
    try:
        result = await improve_resume_service(resume, job_description, job_title, extra_info)
        return result
    except Exception as e:
        logger.error(f"Error occurred while improving resume: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))