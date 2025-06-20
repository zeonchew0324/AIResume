import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
import io
import PyPDF2

load_dotenv()

open_api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="AI Resume Analyzer", page_icon=":guardsman:", layout="centered")

st.title("AI Resume Analyzer")

st.markdown("Welcome to the AI Resume Analyzer! Upload your resume in PDF format, and enjoy your feedback!")