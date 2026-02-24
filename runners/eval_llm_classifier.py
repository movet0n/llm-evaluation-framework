import json
import math
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


def _handle_matrix_nan(value: float, round_precision: int) -> float | None:
    return None if math.isnan(value) else round(value, round_precision)


def outcome(gold: str, pred: str, positive: str = "toxic") -> str:
    gold_pos = gold == positive

    if pred == "needs_review":
        return "NR_toxic" if gold_pos else "NR_safe"

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
    marker = " <-- NEEDS REVIEW (gold: toxic)" if label == "NR_toxic" else marker
    marker = " <-- NEEDS REVIEW (gold: safe)" if label == "NR_safe" else marker
    print(f"  {id:>2}  [{tier:<12}]  gold={gold:<5}  pred={pred:<13}  {conf}  {label}{marker}{extra}")


def print_tier_breakdown(predictions: list) -> dict:
    print("\n--- Per-tier breakdown ---")
    print(
        f"  {'tier':<14}  {'n':>2}  {'TP':>2}  {'FP':>2}  {'TN':>2}  {'FN':>2}  {'recall':>6}  {'precision':>9}  {'f1':>5}  {'specificity':>11}  {'false positive rate':>19}"
    )
    tier_results = {}

    # Iterate through all the tiers ("toxic", "borderline" ... )
    for tier in TIER_ORDER:
        rows = [p for p in predictions if p.get("tier") == tier]
        if not rows:
            continue

        matrix = confusion_from_predictions(rows, positive_label="toxic")  # Only counting happens here
        tier_results[tier] = {  # Now the properties formula runs
            "n": len(rows),
            "tp": matrix.tp,
            "fp": matrix.fp,
            "tn": matrix.tn,
            "fn": matrix.fn,
            "recall": _handle_matrix_nan(matrix.recall, 3),
            "precision": _handle_matrix_nan(matrix.precision, 3),
            "f1": _handle_matrix_nan(matrix.f1, 3),
            "specificity": _handle_matrix_nan(matrix.specificity, 3),
            "false_positive_rate": _handle_matrix_nan(matrix.false_positive_rate, 3),
        }
        print(
            f"  {tier:<14}  {len(rows):>2}  {matrix.tp:>2}  {matrix.fp:>2}  {matrix.tn:>2}  {matrix.fn:>2}  {matrix.recall:>6.3f}  {matrix.precision:>9.3f}  {matrix.f1:>5.3f}  {matrix.specificity:>11.3f}  {matrix.false_positive_rate:>19.3f}"
        )

    return tier_results


def save_report(
    model: str,
    predictions: list,
    matrix,
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
        "confusion_matrix": {"tp": matrix.tp, "fp": matrix.fp, "tn": matrix.tn, "fn": matrix.fn},
        "metrics": {
            "accuracy": _handle_matrix_nan(matrix.accuracy, 4),
            "precision": _handle_matrix_nan(matrix.precision, 4),
            "recall": _handle_matrix_nan(matrix.recall, 4),
            "f1": _handle_matrix_nan(matrix.f1, 4),
            "specificity": _handle_matrix_nan(matrix.specificity, 4),
            "false_positive_rate": _handle_matrix_nan(matrix.false_positive_rate, 4),
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
        # Predictions - a list of the model predictions:
        # [{'id': '1', 'text': "I hope you fail, you're useless.", 'gold': 'toxic', 'tier': 'clear_toxic', 'pred': 'toxic', 'confidence': 1.0} ... ]
        predictions = []
        for row in DATASET:
            pred_label, conf = classify_with_dual_prompt(client, row["text"], model=model)
            out = {**row, "pred": pred_label, "confidence": conf}
            predictions.append(out)
            print_row(row["id"], row.get("tier", ""), row["gold"], pred_label, f"conf={conf:.2f}")

    matrix = confusion_from_predictions(predictions, positive_label="toxic")

    fn_rows = [p for p in predictions if outcome(p["gold"], p["pred"]) == "FN"]
    fp_rows = [p for p in predictions if outcome(p["gold"], p["pred"]) == "FP"]

    print("\n--- Confusion matrix ---")
    print(f"  TP={matrix.tp}  FP={matrix.fp}  TN={matrix.tn}  FN={matrix.fn}")

    print("\n--- Metrics ---")
    print(f"  Accuracy : {matrix.accuracy:.3f}")
    print(f"  Precision: {matrix.precision:.3f}")
    print(f"  Recall   : {matrix.recall:.3f}")
    print(f"  F1       : {matrix.f1:.3f}")

    if fn_rows:
        print(f"\n--- False Negatives ({len(fn_rows)}) — toxic missed as safe ---")
        for p in fn_rows:
            print(f"  [{p['id']:>2}] [{p.get('tier', ''):12}]  \"{p['text'][:70]}\"")

    if fp_rows:
        print(f"\n--- False Positives ({len(fp_rows)}) — safe flagged as toxic ---")
        for p in fp_rows:
            print(f"  [{p['id']:>2}] [{p.get('tier', ''):12}]  \"{p['text'][:70]}\"")

    nr_toxic_rows = [p for p in predictions if outcome(p["gold"], p["pred"]) == "NR_toxic"]
    nr_safe_rows = [p for p in predictions if outcome(p["gold"], p["pred"]) == "NR_safe"]

    if nr_toxic_rows:
        print(f"\n--- Needs Review: uncertain on toxic ({len(nr_toxic_rows)}) — model hedged on actually-toxic content ---")
        for p in nr_toxic_rows:
            print(f"  [{p['id']:>2}] [{p.get('tier', ''):12}]  \"{p['text'][:70]}\"")

    if nr_safe_rows:
        print(f"\n--- Needs Review: uncertain on safe ({len(nr_safe_rows)}) — model unnecessarily hesitant ---")
        for p in nr_safe_rows:
            print(f"  [{p['id']:>2}] [{p.get('tier', ''):12}]  \"{p['text'][:70]}\"")

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

    report_path = save_report(model, predictions, matrix, stability, agreement_threshold, tier_results)
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
