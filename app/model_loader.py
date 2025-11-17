import spacy
import os
from spacy.language import Language

def load_ner_model(model_path: str) -> Language | None:
    if not os.path.exists(model_path):
        print(f"Lỗi: Không tìm thấy model tại '{model_path}'")
        return None
    try:
        nlp = spacy.load(model_path)
        return nlp
    except Exception as e:
        print(f"Lỗi khi tải model: {e}")
        return None
