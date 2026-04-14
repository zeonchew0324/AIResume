from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.prompts.coverletter_prompt import COVERLETTER_PROMPT
from app.config import get_llm
import asyncio

model = get_llm()

async def cover_letter_chain(job_title: str, job_description: str, resume_text: str, company_name: str, extra_info: str = ""):
    prompt = ChatPromptTemplate.from_template(COVERLETTER_PROMPT)

    chain = (
        prompt
        | model
        | StrOutputParser()
    )

    try:
        result = await asyncio.wait_for(
            chain.ainvoke({
                "job_title": job_title,
                "job_description": job_description,
                "resume": resume_text,
                "company_name": company_name,
                "extra_info": extra_info
            }),
            timeout=20.0
        )

        if not result:
            raise ValueError("No response from the model")

    except Exception as e:
        raise ValueError(f"Error during cover letter generation: {str(e)}")

    return result
