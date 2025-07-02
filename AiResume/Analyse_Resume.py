import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
import io
import PyPDF2

from backend.critique_chain import get_critique_chain

load_dotenv()

open_api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="AI Resume analyser", page_icon=":guardsman:", layout="centered")

st.title("AI Resume Analyser")

st.markdown("Welcome! Upload your resume in PDF format, and enjoy your feedback!")

experience_level = ["Select your experience level", "Student Intern", "Entry Level(0-5 years)", 
                    "Mid Level(5-10 years)", "Senior Level(10-20 years)", "Expert Level(20+ years)"]

col1, col2 = st.columns(2)
with col1:
    job_field = st.text_input("Enter the job title/field you are targeting.", placeholder="e.g. Software Engineering/Fintech")
with col2:
    experience = st.selectbox("Select your experience level", experience_level)

uploaded_file = st.file_uploader("Upload your resume (PDF format)", type=["pdf"])

analyse_button = st.button("Analyse Resume")

def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    pdf_text = ""
    for page in pdf_reader.pages:
        pdf_text += page.extract_text() + "\n"
    return pdf_text

def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        return extract_text_from_pdf(io.BytesIO(uploaded_file.read()))
    return uploaded_file.read().decode("utf-8")

if analyse_button and uploaded_file is not None:
    if not job_field or experience == experience_level[0]:
        st.error("Please fill in the job title/field and select your experience level before uploading your resume.")
        st.stop()
    try:
        file_content = extract_text_from_file(uploaded_file)
        if not file_content.strip():
            st.error("The uploaded file is empty or could not be read.")

        analysing_placeholder = st.empty()
        analysing_placeholder.markdown("Analyzing your resume...")

        #Call backend critique chain
        agent = get_critique_chain()
        response = agent.invoke({
            "input": file_content,
            "job_field": job_field,
            "experience_level": experience
        })

        analysing_placeholder.empty()
        feedback = response["output"]
        st.markdown("### 🧠 AI Feedback")
        st.write(feedback)

        # Save feedback in session state
        st.session_state["previous_feedback"] = feedback
        st.session_state["analysed_resume"] = file_content
        st.session_state["job_field"] = job_field
        st.session_state["experience_level"] = experience

        # Display copy-friendly input
        st.markdown("### 📋 Copy This Feedback")
        st.text_area("Copy", feedback, height=250)

        st.info("You can now go to the **Compare Resumes** page in the sidebar and paste this feedback.")

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")


if analyse_button and uploaded_file is None:
    st.warning("No resume found.")