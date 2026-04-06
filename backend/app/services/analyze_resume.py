from fastapi import UploadFile
from app.models.ats import ResumeAnalysisResponse
from app.utils.pdf_parser import extract_text_from_pdf
from app.graphs.ats_graph import ats_chain

async def analyze_resume_service(resume: UploadFile, job_description: str, job_title: str):
    resume_content = extract_text_from_pdf(resume.file)

    if not resume_content:
        raise ValueError("Resume is empty or unreadable")

    result = await ats_chain(job_title=job_title, job_description=job_description, resume_text=resume_content)

    return ResumeAnalysisResponse(**result)