import torch
from app.llm.llm_model_loader import get_llm_model
from app.common.utils import safe_json_parse

# Load model một lần
model, tokenizer = get_llm_model()

# Các field JD mà bạn muốn trích xuất
FIXED_KEYS_JD = [
    "JOB_TITLE",
    "COMPANY",
    "LOCATION",
    "EXPERIENCE_REQUIRED",
    "EDUCATION_REQUIRED",
    "SKILL_Programming_Languages",
    "SKILL_Cloud_DevOps",
    "SKILL_Databases",
    "SKILL_Tools_Frameworks",
    "SKILL_Soft_Skills",
    "RESPONSIBILITIES",
    "BENEFITS",
    "GPA"
]

SYSTEM_PROMPT_JD = f"""
You are an expert NER extractor for Job Descriptions (JD).
Extract exactly the following fields: {', '.join(FIXED_KEYS_JD)}.
Return ONLY a JSON dictionary with these keys.
If a field is missing, return "Không có thông tin".
"""

def llm_extract_ner_jd(text: str):
    try:
        user_prompt = f"JD Text:\n{text}\n\nReturn ONLY JSON with the fixed keys above."
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_JD},
            {"role": "user", "content": user_prompt}
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        output_ids = model.generate(input_ids=input_ids, max_new_tokens=512, do_sample=False)
        text_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        start = text_output.find("{")
        end = text_output.rfind("}") + 1
        json_text = text_output[start:end] if start != -1 and end != -1 else None

        data = safe_json_parse(json_text) if json_text else {}

        entities = {key: data.get(key, "Không có thông tin") for key in FIXED_KEYS_JD}
        return entities

    except Exception as e:
        entities = {key: "Không có thông tin" for key in FIXED_KEYS_JD}
        return {**entities, "error": str(e)}