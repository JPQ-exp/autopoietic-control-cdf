"""
compiler/integrator.py — Library of pure mathematical coordinate mapping utilities.
"""
import numpy as np

def lambert_w0(z):
    """Numerically solves the principal branch of the Lambert W function."""
    if z <= 0.0: 
        return 0.0
    w = np.log1p(z) if z > 1.0 else z * (1.0 - z + 1.5 * z * z)
    for _ in range(15):
        ew = np.exp(w)
        f = w * ew - z
        w -= f / (ew * (w + 1.0) - (w + 2.0) * f / (2.0 * w + 2.0))
    return w


def exact_linear_step(g, rate, forcing, dt):
    """Closed-form exponential integrator in G-space."""
    if abs(rate) < 1e-9:
        return g + forcing * dt
    e = np.exp(rate * dt)
    return g * e + forcing * (e - 1.0) / rate


def ratio_bound_forward(x, x_max):
    """Maps bounded interval [0, x_max) to linear unbounded G-space [0, inf)."""
    safe_x = min(x, x_max - 1e-4)
    return safe_x / (x_max - safe_x)


def ratio_bound_inverse(g, x_max):
    """Maps unbounded G-space [0, inf) back to bounded interval [0, x_max)."""
    g_safe = max(g, 1e-9)
    return x_max * g_safe / (1.0 + g_safe)