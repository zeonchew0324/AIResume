import streamlit as st
import io
import PyPDF2
from backend.comparison import get_comparison_graph

st.set_page_config(page_title="Compare Resumes", layout="centered")

st.title("Compare Resumes")

uploaded_file = st.file_uploader("Upload your old resume (PDF format)", type=["pdf"])

newly_uploaded_file = st.file_uploader("Upload your new resume (PDF format)", type=["pdf"])

previous_feedback = st.text_area(
    "Paste your previous feedback here",
    value=st.session_state.get("previous_feedback", "")
)

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
    return "\n".join([page.extract_text() for page in pdf_reader.pages])

if st.button("Compare My Resumes!") and uploaded_file and newly_uploaded_file:
    try:
        old_resume_text = extract_text_from_pdf(uploaded_file)
        new_resume_text = extract_text_from_pdf(newly_uploaded_file)

        if not old_resume_text.strip() or not new_resume_text.strip():
            st.error("One or both of the uploaded files are empty or could not be read.")
            st.stop()

        st.markdown("Comparing your resumes...")

        # Call backend comparison graph
        comparison_graph = get_comparison_graph()
        result = comparison_graph.invoke({
            "old_resume": old_resume_text,
            "new_resume": new_resume_text,
            "previous_feedback": previous_feedback
        })

        st.markdown("### 📝 Comparison Results")
        st.write(result["output"])

    except Exception as e:
        st.error(f"An error occurred while processing the resumes: {e}")