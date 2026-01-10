from utils.register import register_eval
import numpy as np

from utils.logger import get_logger
logger = get_logger(__name__)

def evaluate(preds, gts, topk=20):
    """
    Computes the Recall for the predictions.
    preds : [batch, n_items] as the probability of each item
    gts : [batch]
    """
    assert len(preds) == len(gts), "Predictions and ground truths must have the same length."
    assert topk > 0, "Top-k must be a positive integer."

    n = len(preds)
    recall = np.zeros(n, dtype=np.float32)

    for i in range(n):
        pred = preds[i]
        gt = gts[i]

        # Get the indices of the top-k predictions
        topk_indices = np.argsort(pred)[-topk:]

        # Check if the ground truth is in the top-k predictions
        if gt in topk_indices:
            recall[i] = 1.0

    return np.mean(recall).item()

@register_eval("ReCall1")
def recall1(preds, gts, topk=1):
    return evaluate(preds, gts, topk=topk)

@register_eval("ReCall5")
def recall5(preds, gts, topk=5):
    return evaluate(preds, gts, topk=topk)

@register_eval("ReCall10")
def recall10(preds, gts, topk=10):
    return evaluate(preds, gts, topk=topk)

@register_eval("ReCall20")
def recall20(preds, gts, topk=20):
    return evaluate(preds, gts, topk=topk)