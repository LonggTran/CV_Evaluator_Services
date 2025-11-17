from unsloth import FastLanguageModel
from transformers import TextStreamer
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models/llm")

# Load model LLM đã fine-tune
def load_llm_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_PATH,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )
    return model, tokenizer


def improve_cv_with_llm(cv_ner: dict, jd_ner: dict, model, tokenizer):
    """
    Gửi lời nhắc (prompt) vào LLM và nhận phản hồi cải thiện CV.
    """

    system_prompt = (
        "You are an expert resume consultant and ATS optimization specialist. "
        "Your job is to compare CV content with job descriptions, identify gaps, "
        "and provide clear, actionable resume improvement suggestions. "
        "Respond in a concise manner, using short bullet points. "
        "Do NOT invent information; only suggest what the candidate can legitimately add."
    )

    user_prompt = (
        "Dưới đây là dữ liệu NER từ CV và JD.\n\n"
        f"--- CV NER OUTPUT ---\n{cv_ner}\n\n"
        f"--- JD NER OUTPUT ---\n{jd_ner}\n\n"
        "Yêu cầu:\n"
        "1. Xác định điểm mạnh hiện có trong CV.\n"
        "2. Liệt kê kỹ năng hoặc yêu cầu bị thiếu so với JD.\n"
        "3. Với mỗi kỹ năng thiếu, giải thích ngắn lý do nó quan trọng.\n"
        "4. Đưa ra các gợi ý cải thiện CV (gạch đầu dòng).\n"
        "5. Đề xuất danh sách kỹ năng chuẩn hơn cho vị trí.\n"
        "6. Trả lời bằng tiếng Việt."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    streamer = TextStreamer(tokenizer)

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        streamer=streamer
    )

    text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text_output
