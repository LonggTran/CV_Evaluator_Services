# app/main.py (hoặc file chứa FastAPI app)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.llm.llm_jd_router import router as llm_jd_router
from app.llm.llm_router import router as llm_router
from app.llm.llm_suggestion import router as cv_suggestion_router  # Thêm dòng này
from app.ner.ner_router import router as ner_router
from app.api.chat import router as chat_router

app = FastAPI(title="Resume Extraction API")

# ---------------------------
# 🚀 FIX CORS CHO FRONTEND
# ---------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # hoặc ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---------------------------

# Include routers
app.include_router(ner_router)
app.include_router(llm_router)  # llm ner resume
app.include_router(llm_jd_router)
app.include_router(chat_router)  # llm chatbot
app.include_router(cv_suggestion_router)  # Thêm dòng này - CV suggestion

@app.get("/")
def root():
    return {
        "message": "Resume Extraction API running!",
        "endpoints": {
            "LLM NER CV": "/llm/extract_ner_cv/",
            "LLM NER JD": "/llm/extract_ner_jd/",
            "CV Suggestion": "/llm/suggest_cv_improvements/",
            "spaCy NER": "/ner/extract_ner/"
        }
    }