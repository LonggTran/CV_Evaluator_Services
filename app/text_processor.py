import re

def preprocess_text(text: str) -> str:
    # Nối từ bị ngắt dòng
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    text = re.sub(r"([a-zA-Z0-9.,])\n([a-zA-Z0-9])", r"\1 \2", text)
    # Chuẩn hóa bullet
    text = re.sub(r"[\uf0b7\u2022\u2023\u25cf\u25aa\u25b6▪▫●•·]", "- ", text)
    # Loại bỏ nhiều khoảng trắng
    text = re.sub(r"\s+", " ", text).strip()
    return text

def postprocess_entities(doc):
    cleaned_entities = []
    for ent in doc.ents:
        t = ent.text.strip(" .,;-")
        if t:
            cleaned_entities.append({
                "text": t,
                "label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char
            })
    return cleaned_entities
