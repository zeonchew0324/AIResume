from app.models.schemas import ResumeAnalysisResponse
from app.graphs.ats_graph import ats_chain

async def analyze_resume_service(resume_text: str, job_description: str, job_title: str):
    if not resume_text:
        raise ValueError("Resume is empty or unreadable")

    result = await ats_chain(job_title=job_title, job_description=job_description, resume_text=resume_text)

    return ResumeAnalysisResponse(**result)
