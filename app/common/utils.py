def safe_json_parse(text: str):
    import json
    try:
        return json.loads(text)
    except:
        return None
