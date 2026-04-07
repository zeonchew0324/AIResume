from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from app.config import MODEL_NAME, OPENAI_API_KEY
from app.prompts.improve_prompt import IMPROVE_PROMPT
from app.config import get_llm

async def improve_resume_chain(job_title: str, job_description: str, resume_text: str, extra_info: str = ""):
    model = get_llm()

    prompt = ChatPromptTemplate.from_template(IMPROVE_PROMPT)

    chain = (
        prompt
        | model
        | JsonOutputParser()
    )

    try:
        result = await chain.ainvoke({
            "job_title": job_title,
            "job_description": job_description,
            "resume_text": resume_text,
            "extra_info": extra_info
        })

        if not result:
            raise ValueError("No response from the model")
        
    except Exception as e:
        raise ValueError(f"Error during resume improvement: {str(e)}")

    return result