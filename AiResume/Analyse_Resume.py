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
    try:
        file_content = extract_text_from_file(uploaded_file)
        if not file_content.strip():
            st.error("The uploaded file is empty or could not be read.")

        analysing_placeholder = st.empty()
        analysing_placeholder.markdown("Analyzing your resume...")

        #Call backend critique chain
        agent = get_critique_chain()
        response = agent.invoke({
            "input": f"Please analyse this resume: \n\n{file_content}"
        })

        analysing_placeholder.empty()
        feedback = response["output"]
        st.markdown("### 🧠 AI Feedback")
        st.write(feedback)

        # Save feedback in session state
        st.session_state["previous_feedback"] = feedback

        # Display copy-friendly input
        st.markdown("### 📋 Copy This Feedback")
        st.text_area("Copy", feedback, height=250)

        st.info("You can now go to the **Compare Resumes** page in the sidebar and paste this feedback.")

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")


if analyse_button and uploaded_file is None:
    st.warning("No resume found.")