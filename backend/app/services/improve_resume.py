from fastapi import UploadFile, File, Form
from app.models.ats import ResumeAnalysisResponse, ResumeImprovementResponse
from app.utils.pdf_parser import extract_text_from_pdf
from app.graphs.improve_graph import improve_resume_chain

async def improve_resume_service(resume: UploadFile = File(...), job_description: str = Form(...), job_title: str = Form(...), extra_info: str = Form("")):
    resumeContent = extract_text_from_pdf(resume.file)

    if not resumeContent:
        raise ValueError("Resume is empty or unreadable")
    
    result = improve_resume_chain(job_title=job_title, job_description=job_description, resume_text=resumeContent, extra_info=extra_info)
    # print("Improve Chain Result:", result)

    return ResumeImprovementResponse(**result)