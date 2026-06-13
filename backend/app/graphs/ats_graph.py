from langchain_core.prompts import ChatPromptTemplate
from app.config import get_llm
from app.models.schemas import ScoreBreakdown
from pydantic import BaseModel
from typing import List
from app.prompts.ats_prompt import NODE_1_EXTRACTION_PROMPT, NODE_2_SYNTHESIS_PROMPT
import asyncio


class Suggestion(BaseModel):
    focus_area: str
    advice: str


class ExtractionResponse(BaseModel):
    score_breakdown: List[ScoreBreakdown]
    missing_keywords: List[str]


class SynthesisResponse(BaseModel):
    match_score: float
    feedback: str
    suggestions: List[Suggestion]


async def ats_chain(job_title: str, job_description: str, resume_text: str) -> dict:
    model = get_llm()

    prompt_1 = ChatPromptTemplate.from_template(NODE_1_EXTRACTION_PROMPT)
    prompt_2 = ChatPromptTemplate.from_template(NODE_2_SYNTHESIS_PROMPT)

    node_1 = prompt_1 | model.with_structured_output(ExtractionResponse)
    node_2 = prompt_2 | model.with_structured_output(SynthesisResponse)

    # Don't catch-and-wrap here: let the real failure (OpenAI error,
    # asyncio.TimeoutError, etc.) propagate with its own type so the route can
    # map it to the correct HTTP status. Converting everything to ValueError
    # would make genuine server failures look like client (400) errors.

    # Node 1: extract the per-category breakdown and missing keywords.
    extracted: ExtractionResponse = await asyncio.wait_for(
        node_1.ainvoke({
            "job_title": job_title,
            "job_description": job_description,
            "resume_text": resume_text,
        }),
        timeout=30.0,  # seconds
    )

    # Node 2: synthesise the final score, feedback, and suggestions from
    # the extraction report.
    synthesis: SynthesisResponse = await asyncio.wait_for(
        node_2.ainvoke({
            "job_title": job_title,
            "extracted_data": extracted.model_dump_json(),
        }),
        timeout=30.0,  # seconds
    )

    # Merge both nodes' outputs into the shape ResumeAnalysisResponse expects.
    return {
        "match_score": synthesis.match_score,
        "feedback": synthesis.feedback,
        "suggestions": [s.model_dump() for s in synthesis.suggestions],
        "missing_keywords": extracted.missing_keywords,
        "score_breakdown": [s.model_dump() for s in extracted.score_breakdown],
    }
