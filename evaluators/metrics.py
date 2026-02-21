from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Metrics:
    tp: int  # TP (True Positive): text IS toxic, model correctly said toxic
    fp: int  # TN (True Negative): text IS safe, model correctly said safe
    tn: int  # FP (False Positive): text IS safe, but model wrongly said toxic - a false alarm
    fn: int  # FN (False Negative): text IS toxic, but model wrongly said safe — a miss (the dangerous one)

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
