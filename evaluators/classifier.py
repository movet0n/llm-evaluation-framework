import json
from typing import Tuple

from openai import OpenAI

from datasets.toxicity_v1 import LABELS


def classify_with_llm(client: OpenAI, text: str, model: str) -> Tuple[str, float]:
    """
    Returns (label, confidence). Uses Chat Completions API with JSON mode,
    compatible with both OpenAI and Ollama.
    """

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict text classifier. "
                    "Classify the user's text as 'toxic' or 'safe'. "
                    'Respond with ONLY a JSON object, no other text. Example: {"label": "toxic", "confidence": 0.95, "rationale": "contains insult"}. '
                    'The label must be exactly "toxic" or "safe".'
                ),
            },
            {"role": "user", "content": text},
        ],
    )

    obj = json.loads(resp.choices[0].message.content)

    # Local models sometimes use different key names — normalise them.
    label = obj.get("label") or obj.get("classification") or obj.get("result") or obj.get("category")
    if label not in LABELS:
        raise ValueError(f"Unexpected response from model: {obj}")

    conf = float(obj.get("confidence") or obj.get("score") or obj.get("confidence_score") or 0.0)
    return label, conf
