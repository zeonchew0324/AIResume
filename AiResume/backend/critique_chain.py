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
        # Extract resume text
        if ":" in input_dict["input"]:
            resume_text = input_dict["input"].split("Please analyse this resume:")[1].strip()
        else:
            resume_text = input_dict["input"]
            
        # Extract job field
        job_field = "Not specified"
        if "job_field" in input_dict and input_dict["job_field"]:
            job_field = input_dict["job_field"]
        elif "targeting " in input_dict["input"]:
            try:
                job_field = input_dict["input"].split("targeting ")[1].split(" positions")[0]
            except:
                pass
                
        # Extract experience level
        experience_level = "Not specified"
        if "experience_level" in input_dict and input_dict["experience_level"]:
            experience_level = input_dict["experience_level"]
        elif "at the " in input_dict["input"]:
            try:
                experience_level = input_dict["input"].split("at the ")[1].split(" experience level")[0]
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