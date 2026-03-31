from fastapi import UploadFile, File, Form
from app.models.ats import ResumeAnalysisResponse
from app.utils.pdf_parser import extract_text_from_pdf
from app.graphs.ats_graph import ats_chain

async def analyze_resume_service(resume: UploadFile = File(...), job_description: str = Form(...), job_title: str = Form(...)):
    resume_content = extract_text_from_pdf(resume.file)

    if not resume_content:
        raise ValueError("Resume is empty or unreadable")
    
    result = ats_chain(job_title=job_title, job_description=job_description, resume_text=resume_content)
    
    return ResumeAnalysisResponse(**result)