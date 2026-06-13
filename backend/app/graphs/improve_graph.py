from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from app.prompts.improve_prompt import IMPROVE_PROMPT
from app.config import get_llm
import asyncio

model = get_llm()


async def improve_resume_chain(job_title: str, job_description: str, resume_text: str, extra_info: str = "") -> dict:
    prompt = ChatPromptTemplate.from_template(IMPROVE_PROMPT)

    chain = (
        prompt
        | model
        | JsonOutputParser()
    )

    # Let the underlying failure propagate with its real type (the route maps
    # asyncio.TimeoutError -> 504 and anything else -> 500).
    result = await asyncio.wait_for(
        chain.ainvoke({
            "job_title": job_title,
            "job_description": job_description,
            "resume_text": resume_text,
            "extra_info": extra_info,
        }),
        timeout=30.0,  # seconds
    )

    if not result:
        # Server-side problem (empty model output), so NOT a ValueError.
        raise RuntimeError("The AI model returned an empty response")

    return result
