import json
from typing import Tuple

from openai import OpenAI

from datasets.toxicity_v1 import LABELS


def classify_with_llm(client: OpenAI, text: str, model: str, temperature: float = 0.0) -> Tuple[str, float]:
    """
    Returns (label, confidence). Uses Chat Completions API with JSON mode,
    compatible with both OpenAI and Ollama.
    Temperature=0 for single deterministic evals.
    Temperature>0 (e.g. 0.7) for stability analysis to surface variability.
    """

    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
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
    label = obj.get("label")
    if label not in LABELS:
        raise ValueError(f"Unexpected model response: {obj}")
    return label, float(obj.get("confidence") or 0.0)
