from fastapi import FastAPI

from app.llm.llm_jd_router import router as llm_jd_router
from app.llm.llm_router import router as llm_router
from app.ner.ner_router import router as ner_router

app = FastAPI(title="Resume Extraction API")

# Include 2 routers
app.include_router(llm_router)
app.include_router(ner_router)
app.include_router(llm_jd_router)
@app.get("/")
def root():
    return {
        "message": "Resume Extraction API running!",
        "endpoints": {
            "LLM NER": "/llm/extract_ner_cv/",
            "LLM NER JD": "/llm/llm_ner_jd/extract_ner_jd/",
            "spaCy NER": "/ner/extract_ner/"
        }
    }
