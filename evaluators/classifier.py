import json
from typing import Tuple

from openai import OpenAI

from datasets.toxicity_v1 import LABELS


def classify_with_llm(client: OpenAI, text: str, model: str, temperature: float = 0.0) -> Tuple[str, float]:
    """
    Returns (label, confidence). Uses Chat Completions API with JSON mode, compatible with both OpenAI and Ollama.
    Temperature=0 for single deterministic evals.
    Temperature>0 (e.g. 0.7) for stability analysis to surface variability.
    """

    # Response example:
    # ChatCompletion(id='chatcmpl-979', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='{"label": "toxic", "confidence": 1.0}', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None))], created=1771765970, model='llama3.2', object='chat.completion', service_tier=None, system_fingerprint='fp_ollama', usage=CompletionUsage(completion_tokens=16, prompt_tokens=106, total_tokens=122, completion_tokens_details=None, prompt_tokens_details=None))
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

    # Resposen: resp.choices[0]                 --> Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='{"label": "toxic", "confidence": 1.0}', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None))
    # Resposen: resp.choices[0].message         --> ChatCompletionMessage(content='{"label": "toxic", "confidence": 1.0}', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None)
    # Resposen: resp.choices[0].message.content --> {"label": "toxic", "confidence": 1.0}
    obj = json.loads(resp.choices[0].message.content)
    label = obj.get("label")
    if label not in LABELS:
        raise ValueError(f"Unexpected model response: {obj}")
    return label, float(obj.get("confidence") or 0.0)
