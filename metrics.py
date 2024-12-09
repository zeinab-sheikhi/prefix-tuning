import evaluate
from typing import List


def compute_blue(refs: List[str], preds: List[str]) -> float:
    sacreblue = evaluate.load("sacrebleu")

    preds = [pred.strip() for pred in preds]
    refs = [[ref.strip()] for ref in refs]

    result = sacreblue.compute(predictions=preds, references=refs)
    return result['score']
