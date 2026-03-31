import io
import PyPDF2

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
    return "\n".join([page.extract_text() for page in pdf_reader.pages])