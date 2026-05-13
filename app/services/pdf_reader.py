import fitz  # PyMuPDF
import re


def extract_and_clean_pdf(pdf_bytes: bytes) -> str:
    """
    Đọc text từ file PDF (dạng bytes) và làm sạch khoảng trắng.
    """
    try:
        # Mở file PDF từ luồng bytes bộ nhớ (không cần lưu xuống ổ cứng)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"

        if not text:
            return ""

        # Làm sạch dữ liệu (Data Cleaning) giống hệt lúc Train trên Kaggle
        text = re.sub(r'[\xa0\t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' +', ' ', text)

        return text.strip()
    except Exception as e:
        print(f"Lỗi đọc PDF: {e}")
        return ""