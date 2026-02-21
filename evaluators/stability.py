import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class SampleStability:
    id: str
    text: str
    gold: str
    predictions: list[str] = field(default_factory=list)
    majority_label: str = ""
    agreement_rate: float = 0.0
    mean_confidence: float = 0.0
    std_confidence: float = 0.0
    unstable: bool = False


def run_stability_analysis(
    rows: list[dict],
    classify_fn: Callable[[str], tuple[str, float]],
    n_runs: int = 5,
    agreement_threshold: float = 0.8,
) -> list[SampleStability]:
    """
    Runs classify_fn n_runs times per sample and measures prediction stability.
    agreement_rate  = fraction of runs that match the majority label.
    mean_confidence = mean confidence score across all runs.
    std_confidence  = standard deviation of confidence scores across runs.
    Samples with agreement_rate < agreement_threshold are flagged as unstable.
    """

    results = []
    for row in rows:
        runs = [classify_fn(row["text"]) for _ in range(n_runs)]
        preds = [label for label, _ in runs]
        confs = [conf for _, conf in runs]

        counter = Counter(preds)
        majority_label, majority_count = counter.most_common(1)[0]
        agreement_rate = round(majority_count / n_runs, 4)

        mean_conf = round(sum(confs) / n_runs, 4)
        variance = sum((c - mean_conf) ** 2 for c in confs) / n_runs
        std_conf = round(math.sqrt(variance), 4)

        results.append(
            SampleStability(
                id=row["id"],
                text=row["text"],
                gold=row["gold"],
                predictions=preds,
                majority_label=majority_label,
                agreement_rate=agreement_rate,
                mean_confidence=mean_conf,
                std_confidence=std_conf,
                unstable=agreement_rate < agreement_threshold,
            )
        )
    return results


def overall_instability(stability: list[SampleStability], agreement_threshold: float = 0.8) -> float:
    """Returns % of samples with agreement_rate below threshold."""
    unstable = sum(1 for s in stability if s.agreement_rate < agreement_threshold)
    return round(unstable / len(stability), 4) if stability else 0.0
