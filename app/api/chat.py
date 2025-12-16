# app/api/chat.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.llm.llm_model_loader import get_llm_model
from app.llm.llm_inference import llm_extract
import torch

router = APIRouter(prefix="/api/llm", tags=["Chat"])


class ChatRequest(BaseModel):
    message: str
    history: list = []


class ChatResponse(BaseModel):
    response: str
    success: bool


# System prompt cho chatbot
CHATBOT_SYSTEM_PROMPT = """
Bạn là trợ lý AI chuyên gia cho hệ thống CVReview - một nền tảng đánh giá và phân tích CV thông minh.

VỀ BẠN:
- Tên: CVReview AI Assistant
- Chuyên môn: Phân tích CV, tư vấn cải thiện hồ sơ, đánh giá JD-CV matching
- Tính cách: Thân thiện, chuyên nghiệp, nhiệt tình hỗ trợ

BẠN CÓ THỂ GIÚP:
1. PHÂN TÍCH CV:
   - Đánh giá điểm mạnh/yếu trong CV
   - Gợi ý cải thiện nội dung CV
   - Tư vấn format CV chuyên nghiệp

2. JD-CV MATCHING:
   - Phân tích mức độ phù hợp CV-JD
   - Gợi ý điều chỉnh CV cho JD cụ thể
   - Đánh giá kỹ năng cần bổ sung

3. HƯỚNG DẪN HỆ THỐNG:
   - Cách upload CV và JD
   - Giải thích kết quả đánh giá
   - Cách đọc biểu đồ điểm số

4. TƯ VẤN NGHỀ NGHIỆP:
   - Xu hướng thị trường việc làm
   - Kỹ năng hot theo ngành
   - Tips phỏng vấn

QUY TẮC:
- Luôn trả lời bằng tiếng Việt (trừ thuật ngữ chuyên môn)
- Cung cấp thông tin chính xác, hữu ích
- Nếu không chắc chắn, hãy nói rõ
- Không đưa thông tin cá nhân người dùng
- Giữ thái độ tích cực, hỗ trợ

Hãy trả lời câu hỏi dựa trên context trên và luôn hữu ích!
"""


@router.post("/chat", response_model=ChatResponse)
async def chat_with_ai(request: ChatRequest):
    try:
        model, tokenizer = get_llm_model()

        # Build conversation history
        messages = [{"role": "system", "content": CHATBOT_SYSTEM_PROMPT}]

        # Add conversation history (last 5 exchanges)
        for msg in request.history[-10:]:  # Last 10 messages for context
            role = "assistant" if msg.get("isBot") else "user"
            messages.append({"role": role, "content": msg.get("text", "")})

        # Add current message
        messages.append({"role": "user", "content": request.message})

        # Tokenize and generate
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        if isinstance(encoded, dict):
            input_ids = encoded["input_ids"].to(model.device)
            attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
            attention_mask = attention_mask.to(model.device)
        else:
            input_ids = encoded.to(model.device)
            attention_mask = torch.ones_like(input_ids).to(model.device)

        # Generate response
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        # Decode response
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Extract only the assistant's response
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()
        elif "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[-1].strip()

        # Clean up any remaining system prompts
        response = response.replace(CHATBOT_SYSTEM_PROMPT[:100], "").strip()

        return ChatResponse(response=response, success=True)

    except Exception as e:
        print(f"Chat error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Xin lỗi, có lỗi xảy ra khi xử lý yêu cầu. Vui lòng thử lại sau."
        )