import json
import os
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from datasets.toxicity_v1 import DATASET
from evaluators.classifier import classify_with_llm
from evaluators.metrics import confusion_from_predictions
from evaluators.stability import overall_instability, run_stability_analysis

load_dotenv()

REPORTS_DIR = Path(__file__).parent.parent / "reports"


def save_report(model: str, predictions: list, m, stability: list | None = None, agreement_threshold: float = 0.8) -> Path:
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

    if stability is not None:
        report["stability"] = {
            "n_runs": len(stability[0].predictions) if stability else 0,
            "agreement_threshold": agreement_threshold,
            "overall_instability_pct": overall_instability(stability, agreement_threshold),
            "samples": [
                {
                    "id": s.id,
                    "text": s.text,
                    "gold": s.gold,
                    "predictions": s.predictions,
                    "majority_label": s.majority_label,
                    "agreement_rate": s.agreement_rate,
                    "mean_confidence": s.mean_confidence,
                    "std_confidence": s.std_confidence,
                    "unstable": s.unstable,
                }
                for s in stability
            ],
        }

    path.write_text(json.dumps(report, indent=2))
    return path


def main():
    api_key = os.getenv("OPENAI_API_KEY", "ollama")
    base_url = os.getenv("OLLAMA_BASE_URL")
    client = OpenAI(api_key=api_key, base_url=base_url)

    model = os.getenv("OLLAMA_MODEL", "gpt-5.2")
    n_runs = int(os.getenv("EVAL_RUNS", "5"))
    agreement_threshold = float(os.getenv("EVAL_AGREEMENT_THRESHOLD", "0.8"))

    stability = None

    if n_runs > 1:
        print(f"Running stability analysis: {n_runs} runs per sample (agreement threshold={agreement_threshold:.0%})\n")
        classify_fn = partial(classify_with_llm, client, model=model)
        stability = run_stability_analysis(DATASET, classify_fn, n_runs=n_runs, agreement_threshold=agreement_threshold)

        predictions = []
        for s in stability:
            row = next(r for r in DATASET if r["id"] == s.id)
            out = {**row, "pred": s.majority_label, "confidence": s.mean_confidence}
            predictions.append(out)
            flag = " *** UNSTABLE" if s.unstable else ""
            print(
                f'{s.id}: gold={s.gold} majority={s.majority_label} '
                f'agreement={s.agreement_rate:.0%} '
                f'mean_conf={s.mean_confidence:.2f} std={s.std_confidence:.2f} '
                f'runs={s.predictions}{flag}'
            )
    else:
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

    if stability:
        unstable = [s for s in stability if s.unstable]
        instability_pct = overall_instability(stability, agreement_threshold)
        print(f"\n--- Stability ({n_runs} runs, agreement threshold={agreement_threshold:.0%}) ---")
        print(f"Overall instability : {instability_pct:.0%} of samples with agreement < {agreement_threshold:.0%}")
        print(f"Unstable samples    : {len(unstable)}/{len(stability)}")
        for s in unstable:
            print(f"  [{s.id}] {s.text[:60]}")
            print(f"        agreement={s.agreement_rate:.0%}  mean_conf={s.mean_confidence:.2f}  std={s.std_confidence:.2f}  runs={s.predictions}")

    report_path = save_report(model, predictions, m, stability, agreement_threshold)
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
