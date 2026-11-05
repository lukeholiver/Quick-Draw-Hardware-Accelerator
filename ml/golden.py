"""
golden.py — the integer-only reference model. THE RTL MUST MATCH THIS BIT-FOR-BIT.

Every operation here has a direct hardware equivalent:

    pixels x_q      : unsigned 8-bit, 0..255           (straight from the UART, no preprocessing)
    weights W_q     : signed 8-bit,  -127..127
    biases b_q      : signed 32-bit, already in accumulator scale
    hidden h_q      : unsigned 8-bit, 0..255           (post-ReLU, so never negative)

    Layer 1, per hidden neuron j:
        acc  = b1_q[j]                                 (preload accumulator with bias)
        acc += W1_q[j, i] * x_q[i]   for i in 0..783   (784 MACs: unsigned8 x signed8)
        acc  = max(acc, 0)                             (ReLU: check sign bit)
        h_q[j] = clamp((acc * M0 + 2^(shift-1)) >> shift, 0, 255)   (requantize)

    Layer 2, per class k:
        acc  = b2_q[k]
        acc += W2_q[k, j] * h_q[j]   for j in 0..127   (128 MACs: same unsigned8 x signed8)
        score[k] = acc                                 (no ReLU, no requantize)

    pred = argmax(score)   — FIRST index wins on ties (hardware: update only on strictly greater)

Note both layers use the SAME multiply shape (unsigned 8-bit activation x signed 8-bit
weight), so one MAC design serves both layers.

All math uses int64 in numpy so nothing can silently overflow here; the bit-width checks
in quantize.py tell you how wide the hardware registers actually need to be.
"""

import numpy as np


def requantize(acc, M0, shift):
    """
    Scale a non-negative int accumulator down to uint8 0..255.

    Mathematically:  h = round(acc * M)  where M = M0 / 2^shift  is a real number < 1.
    Hardware can't multiply by a fraction, so: multiply by integer M0, then shift right.
    Adding 2^(shift-1) first makes the shift round-to-nearest instead of truncating.
    acc >= 0 here (ReLU already ran), so the shift has no negative-number rounding quirks.
    """
    out = (acc * M0 + (1 << (shift - 1))) >> shift
    return np.clip(out, 0, 255)


def golden_forward(x_q, q):
    """
    x_q : (784,) or (B, 784) integers 0..255
    q   : dict with int arrays W1, b1, W2, b2 and ints M0, shift

    Returns a dict of every intermediate, so testbenches can check each stage:
        acc1  : layer-1 accumulators BEFORE ReLU   (B, 128)
        h     : hidden activations, uint8 range    (B, 128)
        acc2  : layer-2 accumulators = scores      (B, 10)
        pred  : predicted class index              (B,)
    """
    x = np.asarray(x_q, dtype=np.int64)
    W1 = q["W1"].astype(np.int64)
    W2 = q["W2"].astype(np.int64)

    acc1 = x @ W1.T + q["b1"]                         # bias preload + 784 MACs per neuron
    h = requantize(np.maximum(acc1, 0), q["M0"], q["shift"])
    acc2 = h @ W2.T + q["b2"]                         # bias preload + 128 MACs per class
    pred = np.argmax(acc2, axis=-1)                   # numpy also returns the first max

    return {"acc1": acc1, "h": h, "acc2": acc2, "pred": pred}