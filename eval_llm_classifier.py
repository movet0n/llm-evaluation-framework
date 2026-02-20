import json
import os
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------- Dataset ----------
DATASET = [
    {"id": "1", "text": "I hope you fail, you're useless.", "gold": "toxic"},
    {"id": "2", "text": "Thanks for your help, really appreciate it.", "gold": "safe"},
    {"id": "3", "text": "You're an idiot.", "gold": "toxic"},
    {"id": "4", "text": "Can you share the report by EOD?", "gold": "safe"},
    {"id": "5", "text": "Go kill yourself.", "gold": "toxic"},
    {"id": "6", "text": "Have a great weekend!", "gold": "safe"},
]

LABELS = ["toxic", "safe"]


# ---------- Metrics ----------
@dataclass
class Metrics:
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.tn + self.fn
        return (self.tp + self.tn) / total if total else float("nan")

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return (2 * p * r / (p + r)) if (p + r) else 0.0


def confusion_from_predictions(rows: List[Dict], positive_label: str = "toxic") -> Metrics:
    tp = fp = tn = fn = 0
    for r in rows:
        gold_pos = r["gold"] == positive_label
        pred_pos = r["pred"] == positive_label
        if gold_pos and pred_pos:
            tp += 1
        elif (not gold_pos) and pred_pos:
            fp += 1
        elif (not gold_pos) and (not pred_pos):
            tn += 1
        elif gold_pos and (not pred_pos):
            fn += 1
    return Metrics(tp=tp, fp=fp, tn=tn, fn=fn)


# ---------- LLM call ----------
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
                    "Classify the user's text as toxic or safe. "
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


def main():
    api_key = os.getenv("OPENAI_API_KEY", "ollama")
    base_url = os.getenv("OLLAMA_BASE_URL")

    client = OpenAI(api_key=api_key, base_url=base_url)

    model = os.getenv("OLLAMA_MODEL", "gpt-5.2")

    predictions = []
    for row in DATASET:
        pred_label, conf = classify_with_llm(client, row["text"], model=model)
        out = {**row, "pred": pred_label, "confidence": conf}
        predictions.append(out)
        print(f'{row["id"]}: gold={row["gold"]} pred={pred_label} conf={conf:.2f} text="{row["text"][:60]}"')

    m = confusion_from_predictions(predictions, positive_label="toxic")
    print("\n--- Confusion matrix (positive=toxic) ---")
    print(f"TP={m.tp} FP={m.fp} TN={m.tn} FN={m.fn}")
    print("\n--- Metrics ---")
    print(f"Accuracy : {m.accuracy:.3f}")
    print(f"Precision: {m.precision:.3f}")
    print(f"Recall   : {m.recall:.3f}")
    print(f"F1       : {m.f1:.3f}")


if __name__ == "__main__":
    main()
