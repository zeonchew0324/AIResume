from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from landchain_core.runnables import RunnablePassthroug
from langgraph.graph import StateGraph, END

load_dotenv()

def get_comparison_graph():
    #Define model
    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0
    )

    #Define State
    class GraphState:
        old_resume: str
        new_resume: str
        previous_feedback: str
        current_feedback: str = None
        improvements: str = None
        recommendations: str = None
        output: str = None

    workflow = StateGraph(GraphState)

    #Node 1: Analyze improvements from old to new resume
    def analyze_improvements(state):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert resume analyst. Please analyze the improvements made from the old resume to the new resume."),
            ("human", """
            Old Resume:
            {old_resume}

            New Resume:
            {new_resume}

            Previous Feedback:
            {previous_feedback}

            Please analyze what specific changes were made and how they address the previous feedback.
            Focus on meaningful improvements rather than formatting or minor text changes.
            """)
        ])

        chain = prompt | model | StrOutputParser()

        analysis = chain.invoke({
            "old_resume": state.old_resume,
            "new_resume": state.new_resume,
            "previous_feedback": state.previous_feedback
        })

        return {"analysis": analysis}
    
    #Node 2 : Evaluate effectiveness of improvements 
    def evaluate_effectiveness(state):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert resume analyst. Please evaluate the effectiveness of the improvements made in the new resume."),
            ("human", """
            Old Resume:
            {old_resume}
             
            New Resume:
            {new_resume}

            Previous Feedback:
            {previous_feedback}
             
            Improvement Analysis:
            {analysis}
             
            Please provide a detailed evaluation of how well the new resume addresses the previous feedback and improves upon the old resume.
            """)
        ])

        chain = prompt | model | StrOutputParser()

        evaluation = chain.invoke({
            "new_resume": state.new_resume,
            "old_resume": state.old_resume,
            "previous_feedback": state.previous_feedback,
            "improvements": state.improvements,
        })

        return {"evaluation": evaluation}

    #Node 3: Generate recommendations for further improvement
    def generate_recommendations(state):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert resume analyst. Based on the evaluation of the new resume, please provide recommendations for further improvement."),
            ("human", """
            New Resume:
            {new_resume}

            Current Feedback:
            {current_feedback}

            Please provide 3-5 specific, actionable recommendations to further improve this resume.
            Each recommendation should include:
            1. What to change or add
            2. Why this change is important
            3. How to implement the change (with an example if possible)
            """)
        ])

        chain = prompt | model | StrOutputParser()

        recommendations = chain.invoke({
            "new_resume": state.new_resume,
            "current_feedback": state.current_feedback
        })

        return {"recommendations": recommendations}
