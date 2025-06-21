from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
load_dotenv()

def get_critique_chain():
    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert resume analyst. Please analyze the following resume and provide feedback on its strengths and weaknesses."),
        ("human", """
        The user uploading this resume may be:
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

    chain = (
        {"resume_text": lambda x: x["input"].split("Please analyze this resume:")[1].strip()} 
        | prompt 
        | model 
        | StrOutputParser()
    )

    class ChainWrapper:
        def invoke(self, input_dict):
            result = chain.invoke(input_dict)
            return {"output": result}
        
    return ChainWrapper()