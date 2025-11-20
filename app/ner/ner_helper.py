def postprocess_entities(doc):
    entities = {}
    for ent in doc.ents:
        label = ent.label_
        text = ent.text.strip()

        if label not in entities:
            entities[label] = []
        entities[label].append(text)

    return entities


def postprocess_ner_final(entity_dict: dict):
    final = {}
    for label, values in entity_dict.items():
        unique = list(set([v.strip() for v in values if v.strip()]))
        final[label] = unique
    return final
