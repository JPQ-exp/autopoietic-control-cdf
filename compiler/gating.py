"""
compiler/gating.py — Pure gating and decision-gating mathematics.
"""
import numpy as np

def softplus(x):
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def competitive_gate(w, reactive_mask, gamma=2.5):
    """Lateral-inhibition gate. Gamma scales the inhibition sharpness."""
    psi_t = 1.0 / (1.0 + np.exp(-2.0 * w))
    pool_total = psi_t[~reactive_mask].sum()
    psi = np.empty_like(w)
    for i in range(len(w)):
        if reactive_mask[i]:
            psi[i] = softplus(psi_t[i])
        else:
            # We scale the negative pool inhibition by gamma to sharpen selections
            psi[i] = softplus(psi_t[i] - gamma * (pool_total - psi_t[i]))
    return psi