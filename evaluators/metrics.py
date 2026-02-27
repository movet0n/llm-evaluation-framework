from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Metrics:
    tp: int  # TP (True Positive): Text IS toxic, model correctly said toxic
    fp: int  # TN (True Negative): Text IS safe, model correctly said safe
    tn: int  # FP (False Positive): Text IS safe, but model wrongly said toxic - a false alarm
    fn: int  # FN (False Negative): Text IS toxic, but model wrongly said safe — a miss (the dangerous one)

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.tn + self.fn
        return (self.tp + self.tn) / total if total else float("nan")

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else float("nan")

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return (2 * p * r / (p + r)) if (p + r) else float("nan")

    # Of all the actually safe texts, what fraction did the model correctly leave alone?
    @property
    def specificity(self) -> float:
        denom = self.tn + self.fp
        return self.tn / denom if denom else float("nan")

    # Of all the actually safe texts, what fraction did the model wrongly flag as toxic?
    @property
    def false_positive_rate(self) -> float:
        denom = self.fp + self.tn
        return self.fp / denom if denom else float("nan")


def confusion_from_predictions(
    rows: List[Dict],
    positive_label: str = "toxic",
) -> Metrics:
    """
    rows --> list of predictions of the same tier, e.g., [{'id': '1', 'text': "I hope you fail, you're useless.", 'gold': 'toxic', 'tier': 'clear_toxic', 'pred': 'toxic', 'confidence': 0.95} ... ]
    """

    tp = fp = tn = fn = 0

    for r in rows:
        if r["pred"] == "needs_review":
            continue  # excluded from automated metrics; counted separately in the runner

        gold_positive = r["gold"] == positive_label
        pred_positive = r["pred"] == positive_label

        # Here we simply sum all the predictions to the Metrics dataclass
        if gold_positive and pred_positive:
            tp += 1
        elif (not gold_positive) and pred_positive:
            fp += 1
        elif (not gold_positive) and (not pred_positive):
            tn += 1
        elif gold_positive and (not pred_positive):
            fn += 1

    return Metrics(tp=tp, fp=fp, tn=tn, fn=fn)
