from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

def get_critique_chain():
    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert resume analyst. Please analyze the following resume and provide feedback on its strengths and weaknesses."),
        ("human", """
        The user uploading this resume:
        - Is targeting a specific job title or field in: {job_field}
        - Has the following experience level: {experience_level}
        - A university student struggling to get their first internship
        - Someone who feels like they "have nothing" impressive to put on a resume yet
        - A candidate who may have done one or two projects or side gigs, but lacks confidence
        - Someone who’s trying to build a resume that evolves over time to get better opportunities
        
        Your task:
        1. Carefully review the resume content below.
        2. Provide empathetic, constructive, and detailed feedback — highlight what's working and what can be improved.
        3. Tailor your advice to their stage of growth: if it's a beginner resume, guide them on what to do next (e.g. projects, open source, hackathons, etc.).
        4. Break down feedback into sections: Content, Structure, Technical Skills, Experience, Formatting, and Suggestions for Growth.
        
        Resume content:
        {resume_text}
        
        Please provide your analysis in a clear, structured format with specific recommendations.
        """),
    ])

    def parse_input(input_dict):
        resume_text = input_dict.get("input", "")
        job_field = input_dict.get("job_field", "Not specified")
        experience_level = input_dict.get("experience_level", "Not specified")
        
        if job_field == "Not specified" and "targeting" in resume_text:
            try:
                parts = resume_text.split("targeting ")
                if len(parts) > 1:
                    second_part = parts[1]
                    if " positions" in second_part:
                        job_field = second_part.split(" positions")[0]
            except:
                pass
                
        if experience_level == "Not specified" and "at the " in resume_text:
            try:
                parts = resume_text.split("at the ")
                if len(parts) > 1:
                    second_part = parts[1]
                    if " experience level" in second_part:
                        experience_level = second_part.split(" experience level")[0]
            except:
                pass
                
        return {
            "resume_text": resume_text,
            "job_field": job_field,
            "experience_level": experience_level
        }

    chain = (
        parse_input
        | prompt 
        | model 
        | StrOutputParser()
    )

    class ChainWrapper:
        def invoke(self, input_dict):
            result = chain.invoke(input_dict)
            return {"output": result}
        
    return ChainWrapper()