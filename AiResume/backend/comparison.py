from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough
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

        return {"improvements": analysis}

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
            {improvements}
             
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
    
    #Node 4: Compile final output
    def compile_output(state):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert resume analyst. Please compile the final output based on the analysis, evaluation, and recommendations."),
            ("human", """
            Previous Feedback:
            {previous_feedback}

            Improvement Analysis:
            {improvements}

            Evaluation of Effectiveness:
            {current_feedback}

            Recommendations for Further Improvement:
            {recommendations}

            Please compile a final output that summarizes the improvements made, evaluates their effectiveness, and provides actionable recommendations.
            """)
        ])

        chain = prompt | model | StrOutputParser()

        final_output = chain.invoke({
            "previous_feedback": state.previous_feedback,
            "improvements": state.improvements,
            "current_feedback": state.current_feedback,
            "recommendations": state.recommendations
        })

        return {"final_output": final_output}
    
    #Define workflow
    workflow.add_node("analyze_improvements", analyze_improvements)
    workflow.add_node("evaluate_effectiveness", evaluate_effectiveness)
    workflow.add_node("generate_recommendations", generate_recommendations)
    workflow.add_node("compile_output", compile_output)

    workflow.add_edge("analyze_improvements", "evaluate_effectiveness")
    workflow.add_edge("evaluate_effectiveness", "generate_recommendations")
    workflow.add_edge("generate_recommendations", "compile_output")
    workflow.add_edge("compile_output", END)

    workflow.set_entry_point("analyze_improvements")

    graph = workflow.compile()

    class GraphWrapper:
        def invoke(self, input_dict):
            state = GraphState(
                old_resume=input_dict["old_resume"],
                new_resume=input_dict["new_resume"],
                previous_feedback=input_dict["previous_feedback"]
            )
            for event in graph.stream(state):
                pass

            final_state = event.state
            return {"output": final_state.output}
        
    return GraphWrapper()
