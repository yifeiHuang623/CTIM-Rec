from utils.register import register_eval
import numpy as np

from utils.logger import get_logger
logger = get_logger(__name__)

def evaluate(preds, gts, topk=20):
    assert len(preds) == len(gts), "Predictions and ground truths must have the same length."
    assert topk > 0, "Top-k must be a positive integer."

    n = len(preds)
    ndcg = np.zeros(n, dtype=np.float32)

    for i in range(n):
        pred = np.asarray(preds[i])
        gt = int(gts[i])
        assert 0 <= gt < pred.size, f"gt index out of range: {gt}"

        p = 1 + int(np.count_nonzero(pred > pred[gt]))
        if p <= min(topk, pred.size):
            ndcg[i] = 1.0 / np.log2(p + 1)

    return float(np.mean(ndcg))


@register_eval("NDCG1")
def ndcg1(preds, gts, topk=1):
    return evaluate(preds, gts, topk=topk)

@register_eval("NDCG5")
def ndcg5(preds, gts, topk=5):
    return evaluate(preds, gts, topk=topk)

@register_eval("NDCG10")
def ndcg10(preds, gts, topk=10):
    return evaluate(preds, gts, topk=topk)

@register_eval("NDCG20")
def ndcg20(preds, gts, topk=20):
    return evaluate(preds, gts, topk=topk)
