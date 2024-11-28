import sacrebleu
from typing import List


def compute_bleu(refs: List[str], preds: List[str]) -> float:
    formatted_refs = [[ref] for ref in refs]
    
    bleu = sacrebleu.corpus_bleu(preds, formatted_refs)
    
    return bleu.score
