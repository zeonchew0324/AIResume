from fastapi import UploadFile
from app.utils.pdf_parser import extract_text_from_pdf
from app.graphs.coverletter_graph import cover_letter_chain

async def generate_coverletter(resume: UploadFile, job_title: str, job_description: str, company_name: str, extra_info: str = ""):
    resume_content = extract_text_from_pdf(resume.file)

    if not resume_content:
        raise ValueError("Resume is empty or unreadable")

    result = await cover_letter_chain(
        job_title=job_title,
        job_description=job_description,
        resume_text=resume_content,
        company_name=company_name,
        extra_info=extra_info
    )

    return result
