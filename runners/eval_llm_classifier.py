import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from datasets.toxicity_v1 import DATASET
from evaluators.classifier import classify_with_llm
from evaluators.metrics import confusion_from_predictions

load_dotenv()

REPORTS_DIR = Path(__file__).parent.parent / "reports"


def save_report(model: str, predictions: list, m) -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = REPORTS_DIR / f"{timestamp}_{model.replace('/', '-')}.json"

    report = {
        "timestamp": timestamp,
        "model": model,
        "dataset": "toxicity_v1",
        "predictions": predictions,
        "confusion_matrix": {"tp": m.tp, "fp": m.fp, "tn": m.tn, "fn": m.fn},
        "metrics": {
            "accuracy": round(m.accuracy, 4),
            "precision": round(m.precision, 4),
            "recall": round(m.recall, 4),
            "f1": round(m.f1, 4),
        },
    }

    path.write_text(json.dumps(report, indent=2))
    return path


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

    report_path = save_report(model, predictions, m)
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
