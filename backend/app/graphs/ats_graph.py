from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from app.config import MODEL_NAME, OPENAI_API_KEY
from app.prompts.ats_prompt import ATS_PROMPT
import asyncio

async def ats_chain(job_title: str, job_description: str, resume_text: str):
    model = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0,
        api_key=OPENAI_API_KEY
    )

    prompt = ChatPromptTemplate.from_template(ATS_PROMPT)

    chain = (
        prompt
        | model
        | JsonOutputParser()
    )

    try:
        result = await asyncio.wait_for(
            chain.ainvoke({
                "job_title": job_title,
                "job_description": job_description,
                "resume_text": resume_text
            }),
            timeout=20.0 #seconds
        ) 

        if not result:
            raise ValueError("No response from the model")
        
    except Exception as e:
        raise ValueError(f"Error during ATS analysis: {str(e)}")

    return result