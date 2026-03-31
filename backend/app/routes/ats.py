from fastapi import APIRouter
from utils.pdf_parser import extract_text_from_pdf

router = APIRouter()

@router.post("/api/analyze")
async def analyze_resume(
    resume: UploadFile = File(...), 
    job_description: str = Form(...), 
    experience_level: str = Form(...), 
    job_field: str = Form(...)
):
    result = await analyze_resume(resume, job_description, experience_level, job_field)
    return {"output": result}