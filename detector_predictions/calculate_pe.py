""""""
from glob import glob
import pandas as pd
import torch
import warnings


def calculate_pe(y, scores, num_steps: int = 100):
    """ROC AUC over quantile thresholds."""
    # pre-check class availability
    if (y == 1).sum() == 0:
        warnings.warn("no positive (stego) samples, TPR undefined")
    if (y == 0).sum() == 0:
        warnings.warn("no negative (cover) samples, FPR undefined")
    #
    steps = torch.linspace(0.0, 1.0, num_steps + 1, device=y.device, dtype=scores.dtype)
    taus = torch.quantile(scores, steps)

    # compute tpr, fpr for each tau
    cmp = scores[:, None] >= taus[None, :]

    is_pos = (y == 1)[:, None]
    is_neg = (y == 0)[:, None]

    tpr = (cmp & is_pos).float().sum(dim=0) / is_pos.float().sum()
    fpr = (cmp & is_neg).float().sum(dim=0) / is_neg.float().sum()

    # P_E
    pe = 0.5 * ((1 - tpr) + fpr)
    # best index
    idx_best = torch.argmin(pe)
    return pe[idx_best].item()


# iterate model predictions
for model_file in sorted(glob('*.csv')):
    predictions = pd.read_csv(model_file)
    #
    y = torch.from_numpy(predictions['y'].to_numpy().copy())
    logits = torch.from_numpy(predictions[['logit0', 'logit1']].to_numpy())
    assert ~torch.isnan(logits).any(), model_file
    # exit()
    scores = torch.nn.functional.softmax(logits, dim=1)[:, 1]
    #
    pe = calculate_pe(y=y, scores=scores)
    print(model_file, pe)
