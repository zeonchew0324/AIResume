from unittest.mock import Mock, patch

from app.utils.pdf_parser import extract_text_from_pdf


def fake_page(text):
    return Mock(extract_text=Mock(return_value=text))

@patch("app.utils.pdf_parser.PdfReader")
def test_joins_pages_with_newlines(mock_reader):
    mock_reader.return_value.pages = [fake_page("Page one"), fake_page("Page two")]
    assert extract_text_from_pdf(Mock()) == "Page one\nPage two"

@patch("app.utils.pdf_parser.PdfReader")
def test_page_without_text_layer_does_not_crash(mock_reader):
    # Scanned/image-only pages make extract_text() return None
    mock_reader.return_value.pages = [fake_page("Page one"), fake_page(None)]
    assert extract_text_from_pdf(Mock()) == "Page one\n"

@patch("app.utils.pdf_parser.PdfReader")
def test_empty_pdf_returns_empty_string(mock_reader):
    mock_reader.return_value.pages = []
    assert extract_text_from_pdf(Mock()) == ""
