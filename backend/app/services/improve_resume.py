from app.models.schemas import ResumeImprovementResponse
from app.graphs.improve_graph import improve_resume_chain

async def improve_resume_service(resume_text: str, job_description: str, job_title: str, extra_info: str = ""):
    if not resume_text:
        raise ValueError("Resume is empty or unreadable")

    result = await improve_resume_chain(job_title=job_title, job_description=job_description, resume_text=resume_text, extra_info=extra_info)

    return ResumeImprovementResponse(**result)
