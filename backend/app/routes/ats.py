from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.utils.pdf_parser import extract_text_from_pdf
from app.services.ats import analyze_resume_service 
from app.models.ats import ResumeAnalysisResponse

router = APIRouter()

@router.post("/api/analyze")
async def analyze_resume(
    resume: UploadFile = File(...), 
    job_description: str = Form(...), 
    experience_level: str = Form(...), 
    job_field: str = Form(...)
) -> ResumeAnalysisResponse:
    try:
        result = await analyze_resume_service(resume, job_description, experience_level, job_field)
        return {"output": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))