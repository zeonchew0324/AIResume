from app.graphs.coverletter_graph import cover_letter_chain

async def generate_coverletter(resume_text: str, job_title: str, job_description: str, company_name: str, extra_info: str = ""):
    if not resume_text:
        raise ValueError("Resume is empty or unreadable")

    result = await cover_letter_chain(
        job_title=job_title,
        job_description=job_description,
        resume_text=resume_text,
        company_name=company_name,
        extra_info=extra_info
    )

    return result
