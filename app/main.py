from fastapi import FastAPI, UploadFile, File, HTTPException
from app.services.pdf_reader import extract_and_clean_pdf
from app.services.resume_parser import ResumeParserCore
from app.models.schemas import ResumeResponse

app = FastAPI(
    title="AI Resume Parser API",
    description="Hệ thống trích xuất thông tin ứng viên bằng mô hình RoBERTa Fine-tuned.",
    version="1.0"
)

# Khởi tạo model ở cấp độ Toàn cục (Global) để không phải load lại mỗi lần gọi API
ai_parser = ResumeParserCore(model_path="./ner_model")


@app.post("/api/v1/parse-cv", response_model=ResumeResponse)
async def parse_cv_endpoint(file: UploadFile = File(...)):
    """
    API Nhận file PDF CV và trả về định dạng JSON.
    """
    # 1. Kiểm tra định dạng file
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Hệ thống hiện tại chỉ hỗ trợ định dạng PDF.")

    try:
        # 2. Đọc file tải lên dạng bytes
        pdf_bytes = await file.read()

        # 3. Trích xuất text
        text = extract_and_clean_pdf(pdf_bytes)
        if not text:
            raise HTTPException(status_code=400,
                                detail="Không thể đọc được văn bản từ PDF (Có thể là ảnh scan/bảo mật).")

        # 4. Phân tích bằng AI Model
        parsed_data = ai_parser.parse(text)

        # 5. Trả kết quả JSON
        return parsed_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi Server nội bộ: {str(e)}")