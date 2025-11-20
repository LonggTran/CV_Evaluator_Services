import pdfplumber

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    text = ""

    try:
        import io
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
    except Exception as e:
        print("PDF error:", e)
        return ""

    return text.strip()
