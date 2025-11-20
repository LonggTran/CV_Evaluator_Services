import os
from unsloth import FastLanguageModel
import torch
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "llm")

model = None
tokenizer = None

def get_llm_model():
    """
    Trả về model và tokenizer đã load.
    Nếu chưa load thì load lần đầu.
    Không dùng GPU/CPU phân chia tự động.
    Không dùng 4bit.
    """
    global model, tokenizer

    if model is None or tokenizer is None:
        print("🔄 Loading LLM model...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_PATH,   # thư mục model
            max_seq_length=2048,
            dtype=None               # để mặc định (FP32/FP16 tùy môi trường)
        )
        print("✅ Model loaded successfully.")

    return model, tokenizer
