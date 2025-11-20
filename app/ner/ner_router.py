from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from app.common.pdf_parser import extract_text_from_pdf_bytes
from app.common.text_processor import preprocess_text
from .ner_inference import run_spacy_ner

router = APIRouter(prefix="/ner", tags=["spaCy NER"])

@router.post("/extract_ner/")
async def extract_ner(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    raw_text = extract_text_from_pdf_bytes(pdf_bytes)

    if not raw_text:
        return JSONResponse(status_code=400, content={"error": "Không thể đọc PDF."})

    clean_text = preprocess_text(raw_text)
    parsed = run_spacy_ner(clean_text)

    # Convert to expected output format
    final_entities = []
    for label, values in parsed.items():
        for text in values:
            final_entities.append({"label": label, "text": text})

    return {"entities": final_entities}
