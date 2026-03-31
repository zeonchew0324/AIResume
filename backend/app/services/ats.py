from fastapi import UploadFile, File, Form
from app.models.ats import ResumeAnalysisResponse
from app.utils.pdf_parser import extract_text_from_pdf

async def analyze_resume_service(resume: UploadFile = File(...), job_description: str = Form(...), experience_level: str = Form(...), job_field: str = Form(...)):
    resume_content = extract_text_from_pdf(resume.file)

    if not resume_content:
        raise ValueError("Resume is empty or unreadable")
    
    return ResumeAnalysisResponse(score=0.0, feedback=[], suggestions=[], missing_keywords=[])