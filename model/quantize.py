"""Quantise the trained float MLP into the integer constants the hardware uses.

Reads  model/data/float_weights.npz  (written by train.py)
Writes model/data/int8_model.npz     (read by golden.py)

Every integer carries an implicit unit (its scale):  real value = integer x scale.
The scales are only used here, offline. golden.py and the hardware see integers only.
"""
from pathlib import Path

import numpy as np

from dataset import load_split
import mlp

DATA_DIR = Path(__file__).parent / "data"
FLOAT_WEIGHTS_FILE = DATA_DIR / "float_weights.npz"
INT_MODEL_FILE = DATA_DIR / "int8_model.npz"

INT8_MAX = 127        # weights use -127..127 (symmetric), activations use 0..127 (after ReLU)
SHIFT = 16            # requantisation right shift n: the unit-change factor M is stored as m / 2^16
INPUT_SCALE = 1 / 16  # the raw pixels 0..16 are the INT8 inputs; float training used pixels / 16


def load_float_params():
    """Read the float weights saved by train.py."""
    data = np.load(FLOAT_WEIGHTS_FILE)
    params = {}
    for name in ["W1", "b1", "W2", "b2"]:
        params[name] = data[name]
    return params


def quantize_symmetric(values, scale):
    """Convert floats to integers in the given unit: round(values / scale), clipped to -127..127."""
    integers = np.round(values / scale)
    integers = np.clip(integers, -INT8_MAX, INT8_MAX)
    return integers.astype(np.int64)


def quantize_model(params, X_calibration):
    """Turn float params into integer weights, biases and requantisation multipliers.

    X_calibration: float training images (pixels / 16), used to find the largest hidden value.
    """
    num_hidden = params["W1"].shape[0]

    # ---- Layer 1 weights: one scale per hidden neuron (one per row of W1) ----
    # Each row uses its own largest value, so a row of small weights keeps its precision.
    W1 = params["W1"]
    w1_scale = np.max(np.abs(W1), axis=1) / INT8_MAX                 # (32,)
    q_W1 = quantize_symmetric(W1, w1_scale.reshape(num_hidden, 1))   # row i divided by w1_scale[i]

    # Layer 1 accumulator unit = weight unit x input unit (one value per neuron)
    acc1_scale = w1_scale * INPUT_SCALE

    # The bias is added straight onto the accumulator, so it must use the accumulator unit
    b1_int = np.round(params["b1"] / acc1_scale).astype(np.int64)

    # ---- Hidden activations: one unit shared by all 32, from the largest value seen in training ----
    hidden_pre, hidden, logits = mlp.forward(params, X_calibration)
    hidden_scale = np.max(hidden) / INT8_MAX

    # ---- Requantisation: change unit from acc1_scale to hidden_scale ----
    # M = old unit / new unit is a small fraction, so store it as the integer m = round(M x 2^SHIFT).
    # The hardware then computes (acc x m + 2^(SHIFT-1)) >> SHIFT: integer multiply, add, shift.
    M1 = acc1_scale / hidden_scale
    m1 = np.round(M1 * 2**SHIFT).astype(np.int64)

    # ---- Layer 2 weights: ONE scale for the whole matrix ----
    # Then all 10 output scores share the same unit, so the largest integer score is already
    # the predicted digit. The output layer needs no requantisation.
    W2 = params["W2"]
    w2_scale = np.max(np.abs(W2)) / INT8_MAX
    q_W2 = quantize_symmetric(W2, w2_scale)

    acc2_scale = w2_scale * hidden_scale
    b2_int = np.round(params["b2"] / acc2_scale).astype(np.int64)

    model = {
        # Integer constants: the only things golden.py and the hardware use
        "W1": q_W1,
        "b1": b1_int,
        "m1": m1,
        "W2": q_W2,
        "b2": b2_int,
        "shift": np.int64(SHIFT),
        # Float units, kept for reporting only
        "w1_scale": w1_scale,
        "hidden_scale": np.float64(hidden_scale),
        "w2_scale": np.float64(w2_scale),
    }
    return model


if __name__ == "__main__":
    params = load_float_params()

    # Calibrate on training images only: the test set must stay unseen
    X_train, X_test, y_train, y_test = load_split()
    model = quantize_model(params, X_train / 16.0)

    print(f"layer 1 weight units: {np.min(model['w1_scale']):.5f} .. {np.max(model['w1_scale']):.5f}")
    print(f"hidden unit:          {model['hidden_scale']:.5f}")
    print(f"layer 2 weight unit:  {model['w2_scale']:.5f}")
    print(f"m1 range:             {np.min(model['m1'])} .. {np.max(model['m1'])}   (shift = {SHIFT})")
    print(f"max |b1_int| = {np.max(np.abs(model['b1']))},  max |b2_int| = {np.max(np.abs(model['b2']))}")

    np.savez(INT_MODEL_FILE, **model)
    print(f"saved {INT_MODEL_FILE}")
