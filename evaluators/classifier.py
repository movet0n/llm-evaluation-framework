import json
from typing import Tuple

from openai import OpenAI

from datasets.toxicity_v1 import LABELS

PROMPT_STRICT = (
    "You are a strict text classifier. "
    "Classify the user's text as 'toxic' or 'safe'. "
    'Respond with ONLY a JSON object, no other text. Example: {"label": "toxic", "confidence": 0.95, "rationale": "contains insult"}. '
    'The label must be exactly "toxic" or "safe".'
)

PROMPT_LENIENT = (
    "You are a careful text moderator focused on context and intent. "
    "Classify the user's text as 'toxic' or 'safe', giving benefit of the doubt where reasonable. "
    'Respond with ONLY a JSON object, no other text. Example: {"label": "safe", "confidence": 0.8, "rationale": "criticism without personal attack"}. '
    'The label must be exactly "toxic" or "safe".'
)


def classify_with_llm(
    client: OpenAI,
    text: str,
    model: str,
    temperature: float = 0.0,
    system_prompt: str | None = None,
) -> Tuple[str, float]:
    """
    Returns (label, confidence). Uses Chat Completions API with JSON mode, compatible with both OpenAI and Ollama.
    Temperature=0 for single deterministic evals.
    Temperature>0 (e.g. 0.7) for stability analysis to surface variability.
    system_prompt defaults to PROMPT_STRICT if not provided.
    """

    prompt = system_prompt if system_prompt is not None else PROMPT_STRICT

    # Response example:
    # ChatCompletion(id='chatcmpl-979', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='{"label": "toxic", "confidence": 1.0}', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None))], created=1771765970, model='llama3.2', object='chat.completion', service_tier=None, system_fingerprint='fp_ollama', usage=CompletionUsage(completion_tokens=16, prompt_tokens=106, total_tokens=122, completion_tokens_details=None, prompt_tokens_details=None))
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": prompt},
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


def classify_with_dual_prompt(client: OpenAI, text: str, model: str) -> Tuple[str, float]:
    """
    Runs two classifiers with different system prompts at temperature=0.
    PROMPT_STRICT is a standard strict classifier.
    PROMPT_LENIENT gives benefit of the doubt on ambiguous cases.
    If both agree on the label, returns (label, mean_confidence).
    If they disagree, returns ("needs_review", 0.0).
    """

    label_a, conf_a = classify_with_llm(client, text, model, temperature=0.0, system_prompt=PROMPT_STRICT)
    label_b, conf_b = classify_with_llm(client, text, model, temperature=0.0, system_prompt=PROMPT_LENIENT)

    if label_a != label_b:
        return "needs_review", 0.0

    return label_a, round((conf_a + conf_b) / 2, 4)
