from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request, Depends  
from app.services.analyze_resume import analyze_resume_service
from app.services.improve_resume import improve_resume_service 
from app.services.cover_letter import generate_coverletter
from app.models.schemas import CoverLetterResponse, ResumeAnalysisResponse, ResumeImprovementResponse
from app.limiter import limiter
from app.utils.input_cleaner import clean_input, MAX_EXTRA_INFO_LENGTH, MAX_JD_LENGTH, MAX_JOB_TITLE_LENGTH


import logging 
import traceback

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/health")
async def health_check():
    return {"status": "ok"}

@router.post("/api/analyze")
@limiter.limit("5/minute")
async def analyze_resume(
    request: Request,
    resume: UploadFile = File(...), 
    job_description: str = Form(...), 
    job_title: str = Form(...),
) -> ResumeAnalysisResponse:
    try:
        job_description = clean_input(job_description, MAX_JD_LENGTH)
        job_title = clean_input(job_title, MAX_JOB_TITLE_LENGTH)
        result = await analyze_resume_service(resume, job_description, job_title)
        return result
    except ValueError as e:
        logger.error(f"Validation error in analyze: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error occurred while analyzing resume: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Analysis failed. Please try again.")


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
        extra_info = clean_input(extra_info, MAX_EXTRA_INFO_LENGTH, required=False)
        job_description = clean_input(job_description, MAX_JD_LENGTH)
        job_title = clean_input(job_title, MAX_JOB_TITLE_LENGTH)
        result = await improve_resume_service(resume, job_description, job_title, extra_info)
        return result
    except ValueError as e:
        logger.error(f"Validation error in improve: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error occurred while improving resume: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Improvement failed. Please try again.")
    
@router.post("/api/coverletter")
@limiter.limit("5/minute")
async def create_coverletter(
    request: Request,
    resume: UploadFile = File(...),
    job_title: str = Form(...),
    job_description: str = Form(...),
    company_name: str = Form(...),
    extra_info: str = Form("")
) -> CoverLetterResponse:
    try:
        extra_info = clean_input(extra_info, MAX_EXTRA_INFO_LENGTH, required=False)
        job_description = clean_input(job_description, MAX_JD_LENGTH)
        job_title = clean_input(job_title, MAX_JOB_TITLE_LENGTH)
        company_name = clean_input(company_name, MAX_JOB_TITLE_LENGTH)
        cover_letter = await generate_coverletter(resume, job_title, job_description, company_name, extra_info)
        return CoverLetterResponse(cover_letter=cover_letter)
    except ValueError as e:
        logger.error(f"Validation error in create_coverletter: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error occurred while creating cover letter: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Cover letter generation failed. Please try again.")
    
