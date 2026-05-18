from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from app.config import MODEL_NAME, OPENAI_API_KEY
from app.config import get_llm
from pydantic import BaseModel, Field
from typing import List
from langchain_core.runnables import RunnablePassthrough
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

    chain = (
        {
            "extracted_data": node_1, 
            "job_title": RunnablePassthrough() | (lambda x: x["job_title"])
        }
        | node_2
    )

    try:
        result = await asyncio.wait_for(
            chain.ainvoke({
                "job_title": job_title,
                "job_description": job_description,
                "resume_text": resume_text
            }),
            timeout=30.0 #seconds
        ) 

        if not result:
            raise ValueError("No response from the model")
        
    except Exception as e:
        raise ValueError(f"Error during ATS analysis: {str(e)}")

    return result