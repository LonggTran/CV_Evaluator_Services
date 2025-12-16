from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from app.llm.llm_inference import llm_extract

router = APIRouter(prefix="/llm", tags=["LLM JD"])

@router.post("/extract_ner_jd/")
async def extract_ner_jd(request: Request):
    try:
        # đọc bytes và decode UTF-8 (hỗ trợ tiếng Việt)
        raw_bytes = await request.body()
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return JSONResponse(
            status_code=400,
            content={"error": "Không thể decode UTF-8. Kiểm tra encoding client."}
        )

    if not text.strip():
        return JSONResponse(status_code=400, content={"error": "Input text trống."})

    # gọi LLM xử lý
    entities = llm_extract(text, mode="jd")
    return entities