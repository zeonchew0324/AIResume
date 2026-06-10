from langchain_core.prompts import ChatPromptTemplate
from app.config import get_llm
from pydantic import BaseModel
from typing import List
from app.prompts.ats_prompt import NODE_1_EXTRACTION_PROMPT, NODE_2_SYNTHESIS_PROMPT
import asyncio

class ExtractionResponse(BaseModel):
    score_breakdown: List[dict]
    missing_keywords: List[str]

class FinalResponse(BaseModel):
    match_score: float
    feedback: str
    suggestions: List[dict]


async def ats_chain(job_title: str, job_description: str, resume_text: str):
    model = get_llm()

    prompt_1 = ChatPromptTemplate.from_template(NODE_1_EXTRACTION_PROMPT)
    prompt_2 = ChatPromptTemplate.from_template(NODE_2_SYNTHESIS_PROMPT)

    node_1 = prompt_1 | model.with_structured_output(ExtractionResponse)
    node_2 = prompt_2 | model.with_structured_output(FinalResponse)

    async def run_chain():
        extraction = await node_1.ainvoke({
            "job_title": job_title,
            "job_description": job_description,
            "resume_text": resume_text
        })
        final = await node_2.ainvoke({
            "job_title": job_title,
            "extracted_data": extraction.model_dump_json()
        })
        return extraction, final

    try:
        extraction, final = await asyncio.wait_for(run_chain(), timeout=30.0) #seconds

        if not extraction or not final:
            raise ValueError("No response from the model")

    except Exception as e:
        raise ValueError(f"Error during ATS analysis: {str(e)}")

    return {
        **final.model_dump(),
        "missing_keywords": extraction.missing_keywords,
        "score_breakdown": extraction.score_breakdown,
    }