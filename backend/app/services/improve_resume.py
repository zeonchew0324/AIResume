from fastapi import UploadFile
from app.models.ats import ResumeImprovementResponse
from app.utils.pdf_parser import extract_text_from_pdf
from app.graphs.improve_graph import improve_resume_chain

async def improve_resume_service(resume: UploadFile, job_description: str, job_title: str, extra_info: str = ""):
    resume_content = extract_text_from_pdf(resume.file)

    if not resume_content:
        raise ValueError("Resume is empty or unreadable")

    result = await improve_resume_chain(job_title=job_title, job_description=job_description, resume_text=resume_content, extra_info=extra_info)

    return ResumeImprovementResponse(**result)