from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os

from .model_loader import load_ner_model
from .pdf_parser import extract_text_from_pdf_bytes
from .text_processor import preprocess_text, postprocess_entities
from .ner_helper import postprocess_ner_final

app = FastAPI(title="Resume NER API")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model-best-1")
nlp = load_ner_model(MODEL_PATH)

@app.post("/extract_ner/")
async def extract_ner(file: UploadFile = File(...)):
    if not nlp:
        return JSONResponse(status_code=500, content={"error": "Model chưa được tải."})

    # Đọc file PDF từ bytes trực tiếp
    pdf_bytes = await file.read()
    raw_text = extract_text_from_pdf_bytes(pdf_bytes)
    if not raw_text:
        return JSONResponse(status_code=400, content={"error": "Không thể đọc file PDF."})

    clean_text = preprocess_text(raw_text)
    doc = nlp(clean_text)

    # Hậu xử lý NER
    raw_entities = postprocess_entities(doc)
    merged_dict = postprocess_ner_final(raw_entities)

    # Chuẩn hóa JSON trả về Spring Boot
    final_entities = []
    for label, values in merged_dict.items():
        for text in values:
            final_entities.append({
                "label": label,
                "text": text
            })

    return {"entities": final_entities}  # Không cần filename, gọn nhẹ
