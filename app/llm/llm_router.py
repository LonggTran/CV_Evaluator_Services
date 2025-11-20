from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from app.common.pdf_parser import extract_text_from_pdf_bytes
from app.common.text_processor import preprocess_text
from app.llm.llm_inference import llm_extract

router = APIRouter(prefix="/llm", tags=["LLM CV"])

@router.post("/extract_ner_cv/")
async def extract_ner_cv(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    raw_text = extract_text_from_pdf_bytes(pdf_bytes)

    if not raw_text:
        return JSONResponse(status_code=400, content={"error": "Không thể đọc PDF."})

    clean_text = preprocess_text(raw_text)
    entities = llm_extract(clean_text, mode="cv")

    return entities
