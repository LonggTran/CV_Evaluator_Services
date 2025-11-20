import re

def preprocess_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def normalize_missing(value: str):
    return value if value else "Không có thông tin"
