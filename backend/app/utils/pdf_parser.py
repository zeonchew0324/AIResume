import io
import PyPDF2

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    # extract_text() returns None for pages with no text layer (e.g. scans)
    return "\n".join([page.extract_text() or "" for page in pdf_reader.pages])