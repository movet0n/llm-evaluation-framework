import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from datasets.toxicity_v1 import DATASET
from evaluators.classifier import classify_with_dual_prompt, classify_with_llm
from evaluators.metrics import confusion_from_predictions
from evaluators.stability import overall_instability, run_stability_analysis

load_dotenv()

REPORTS_DIR = Path(__file__).parent.parent / "reports"
TIER_ORDER = ["clear_toxic", "mostly_toxic", "borderline", "mostly_safe", "clear_safe"]


def outcome(gold: str, pred: str, positive: str = "toxic") -> str:
    if pred == "needs_review":
        return "NR"

    gold_pos = gold == positive
    pred_pos = pred == positive
    if gold_pos and pred_pos:
        return "TP"
    if not gold_pos and not pred_pos:
        return "TN"
    if not gold_pos and pred_pos:
        return "FP"

    return "FN"  # gold_pos and not pred_pos — missed toxic


def print_row(id: str, tier: str, gold: str, pred: str, conf: str, extra: str = "") -> None:
    label = outcome(gold, pred)
    marker = " <-- MISSED TOXIC" if label == "FN" else ""
    marker = " <-- FALSE ALARM" if label == "FP" else marker
    marker = " <-- NEEDS REVIEW" if label == "NR" else marker
    print(f"  {id:>2}  [{tier:<12}]  gold={gold:<5}  pred={pred:<13}  {conf}  {label}{marker}{extra}")


def print_tier_breakdown(predictions: list) -> dict:
    print("\n--- Per-tier breakdown ---")
    print(
        f"  {'tier':<14}  {'n':>2}  {'TP':>2}  {'FP':>2}  {'TN':>2}  {'FN':>2}  {'recall':>6}  {'precision':>9}  {'f1':>5}"
    )
    tier_results = {}

    # Iterate through all the tiers ("toxic", "borderline" ... )
    for tier in TIER_ORDER:
        rows = [p for p in predictions if p.get("tier") == tier]
        if not rows:
            continue

        m = confusion_from_predictions(rows, positive_label="toxic")
        tier_results[tier] = {
            "n": len(rows),
            "tp": m.tp,
            "fp": m.fp,
            "tn": m.tn,
            "fn": m.fn,
            "recall": round(m.recall, 3),
            "precision": round(m.precision, 3),
            "f1": round(m.f1, 3),
        }
        print(
            f"  {tier:<14}  {len(rows):>2}  {m.tp:>2}  {m.fp:>2}  {m.tn:>2}  {m.fn:>2}  {m.recall:>6.3f}  {m.precision:>9.3f}  {m.f1:>5.3f}"
        )
    return tier_results


def save_report(
    model: str,
    predictions: list,
    m,
    stability: list | None = None,
    agreement_threshold: float = 0.8,
    tier_results: dict | None = None,
) -> Path:
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

    if tier_results is not None:
        report["tier_metrics"] = tier_results

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

    print(f"\n=== Eval: {model} | dataset: toxicity_v1 ===\n")

    if n_runs > 1:
        print(f"Stability mode: {n_runs} runs per sample (agreement threshold={agreement_threshold:.0%})\n")
        stability_temp = float(os.getenv("EVAL_STABILITY_TEMPERATURE", "0.0"))

        classify_fn = lambda text: classify_with_llm(client, text, model=model, temperature=stability_temp)
        stability = run_stability_analysis(
            DATASET,
            classify_fn,
            n_runs=n_runs,
            agreement_threshold=agreement_threshold,
        )

        # Predictions - a list of the model predictions:
        # [{'id': '1', 'text': "I hope you fail, you're useless.", 'gold': 'toxic', 'tier': 'clear_toxic', 'pred': 'toxic', 'confidence': 1.0} ... ]
        predictions = []
        for s in stability:
            row = next(r for r in DATASET if r["id"] == s.id)
            out = {**row, "pred": s.majority_label, "confidence": s.mean_confidence}
            predictions.append(out)
            unstable_flag = "  [UNSTABLE]" if s.unstable else ""
            print_row(
                s.id,
                row.get("tier", ""),
                s.gold,
                s.majority_label,
                f"agreement={s.agreement_rate:.0%}  mean_conf={s.mean_confidence:.2f}  std={s.std_confidence:.2f}",
                extra=unstable_flag,
            )
    else:
        predictions = []
        for row in DATASET:
            pred_label, conf = classify_with_dual_prompt(client, row["text"], model=model)
            out = {**row, "pred": pred_label, "confidence": conf}
            predictions.append(out)
            print_row(row["id"], row.get("tier", ""), row["gold"], pred_label, f"conf={conf:.2f}")

    m = confusion_from_predictions(predictions, positive_label="toxic")

    fn_rows = [p for p in predictions if outcome(p["gold"], p["pred"]) == "FN"]
    fp_rows = [p for p in predictions if outcome(p["gold"], p["pred"]) == "FP"]

    print("\n--- Confusion matrix ---")
    print(f"  TP={m.tp}  FP={m.fp}  TN={m.tn}  FN={m.fn}")

    print("\n--- Metrics ---")
    print(f"  Accuracy : {m.accuracy:.3f}")
    print(f"  Precision: {m.precision:.3f}")
    print(f"  Recall   : {m.recall:.3f}")
    print(f"  F1       : {m.f1:.3f}")

    if fn_rows:
        print(f"\n--- False Negatives ({len(fn_rows)}) — toxic missed as safe ---")
        for p in fn_rows:
            print(f"  [{p['id']:>2}] [{p.get('tier', ''):12}]  \"{p['text'][:70]}\"")

    if fp_rows:
        print(f"\n--- False Positives ({len(fp_rows)}) — safe flagged as toxic ---")
        for p in fp_rows:
            print(f"  [{p['id']:>2}] [{p.get('tier', ''):12}]  \"{p['text'][:70]}\"")

    nr_rows = [p for p in predictions if p["pred"] == "needs_review"]
    if nr_rows:
        print(f"\n--- Needs Review ({len(nr_rows)}) — strict and lenient prompts disagreed ---")
        for p in nr_rows:
            print(f"  [{p['id']:>2}] [{p.get('tier', ''):12}]  gold={p['gold']:<5}  \"{p['text'][:70]}\"")

    tier_results = print_tier_breakdown(predictions)

    if stability:
        unstable = [s for s in stability if s.unstable]
        instability_pct = overall_instability(stability, agreement_threshold)
        print(f"\n--- Stability ({n_runs} runs, threshold={agreement_threshold:.0%}) ---")
        print(f"  Overall instability : {instability_pct:.0%} of samples below threshold")
        print(f"  Unstable samples    : {len(unstable)}/{len(stability)}")
        for s in unstable:
            print(f"  [{s.id:>2}]  agreement={s.agreement_rate:.0%}  runs={s.predictions}")
            print(f'       "{s.text[:70]}"')

    report_path = save_report(model, predictions, m, stability, agreement_threshold, tier_results)
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
