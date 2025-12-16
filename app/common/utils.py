# app/common/utils.py
import json


def safe_json_parse(json_text):
    """Safely parse JSON text with error handling"""
    if not json_text:
        return {}

    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        # Try to fix common JSON issues
        try:
            # Replace single quotes with double quotes
            json_text = json_text.replace("'", '"')
            return json.loads(json_text)
        except:
            return {}