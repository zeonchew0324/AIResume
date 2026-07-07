from fastapi import APIRouter, Form, HTTPException, Request, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.analyze_resume import analyze_resume_service
from app.services.improve_resume import improve_resume_service
from app.services.cover_letter import generate_coverletter
from app.services.resume_service import get_resume_text
from app.db.database import get_db
from app.models.schemas import CoverLetterResponse, ResumeAnalysisResponse, ResumeImprovementResponse
from app.limiter import limiter
from app.auth import get_current_user_id
from app.utils.input_cleaner import clean_input, MAX_EXTRA_INFO_LENGTH, MAX_JD_LENGTH, MAX_JOB_TITLE_LENGTH


import asyncio
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/health")
async def health_check():
    return {"status": "ok"}

@router.post("/api/analyze")
@limiter.limit("5/minute")
async def analyze_resume(
    request: Request,
    resume_id: str = Form(...),
    job_description: str = Form(...),
    job_title: str = Form(...),
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
) -> ResumeAnalysisResponse:
    try:
        job_description = clean_input(job_description, MAX_JD_LENGTH)
        job_title = clean_input(job_title, MAX_JOB_TITLE_LENGTH)
        resume_text = await get_resume_text(db, resume_id, user_id)
        result = await analyze_resume_service(resume_text, job_description, job_title)
        return result
    except ValueError as e:
        logger.warning(f"Validation error in analyze: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except asyncio.TimeoutError:
        logger.warning("Analyze timed out waiting for the AI model")
        raise HTTPException(status_code=504, detail="The AI model took too long. Please try again.")
    except Exception:
        logger.exception("Error occurred while analyzing resume")
        raise HTTPException(status_code=500, detail="Analysis failed. Please try again.")


@router.post("/api/improve")
@limiter.limit("5/minute")
async def improve_resume(
    request: Request,
    resume_id: str = Form(...),
    job_description: str = Form(...),
    job_title: str = Form(...),
    extra_info: str = Form(""),
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
) -> ResumeImprovementResponse:
    try:
        extra_info = clean_input(extra_info, MAX_EXTRA_INFO_LENGTH, required=False)
        job_description = clean_input(job_description, MAX_JD_LENGTH)
        job_title = clean_input(job_title, MAX_JOB_TITLE_LENGTH)
        resume_text = await get_resume_text(db, resume_id, user_id)
        result = await improve_resume_service(resume_text, job_description, job_title, extra_info)
        return result
    except ValueError as e:
        logger.warning(f"Validation error in improve: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except asyncio.TimeoutError:
        logger.warning("Improve timed out waiting for the AI model")
        raise HTTPException(status_code=504, detail="The AI model took too long. Please try again.")
    except Exception:
        logger.exception("Error occurred while improving resume")
        raise HTTPException(status_code=500, detail="Improvement failed. Please try again.")
    
@router.post("/api/coverletter")
@limiter.limit("5/minute")
async def create_coverletter(
    request: Request,
    resume_id: str = Form(...),
    job_title: str = Form(...),
    job_description: str = Form(...),
    company_name: str = Form(...),
    extra_info: str = Form(""),
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
) -> CoverLetterResponse:
    try:
        extra_info = clean_input(extra_info, MAX_EXTRA_INFO_LENGTH, required=False)
        job_description = clean_input(job_description, MAX_JD_LENGTH)
        job_title = clean_input(job_title, MAX_JOB_TITLE_LENGTH)
        company_name = clean_input(company_name, MAX_JOB_TITLE_LENGTH)
        resume_text = await get_resume_text(db, resume_id, user_id)
        cover_letter = await generate_coverletter(resume_text, job_title, job_description, company_name, extra_info)
        return CoverLetterResponse(cover_letter=cover_letter)
    except ValueError as e:
        logger.warning(f"Validation error in create_coverletter: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except asyncio.TimeoutError:
        logger.warning("Cover letter generation timed out waiting for the AI model")
        raise HTTPException(status_code=504, detail="The AI model took too long. Please try again.")
    except Exception:
        logger.exception("Error occurred while creating cover letter")
        raise HTTPException(status_code=500, detail="Cover letter generation failed. Please try again.")
    
