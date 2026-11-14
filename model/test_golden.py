"""Checks for quantize.py and golden.py. Run from the project root with: python -m pytest model -v

Needs model/data/float_weights.npz, so run python model/train.py once first.
"""
import numpy as np

from dataset import load_split
import golden
import mlp
import quantize


def build_models():
    """Float params, the integer model built from them, and the dataset."""
    params = quantize.load_float_params()
    X_train, X_test, y_train, y_test = load_split()
    model = quantize.quantize_model(params, X_train / 16.0)
    return params, model, X_train, X_test, y_test


# ---- requantize(): the worked examples from the lesson ----

def test_requantize_rounds_to_nearest():
    # 705 x 678 / 2^16 = 7.29, rounds to 7
    assert golden.requantize(np.array([705]), 678, 16)[0] == 7
    # exactly 2.5 rounds up to 3
    assert golden.requantize(np.array([5]), 32768, 16)[0] == 3


def test_requantize_negative_becomes_zero():
    # -300 x 678 / 2^16 = -3.1, and the clamp at 0 is the ReLU
    assert golden.requantize(np.array([-300]), 678, 16)[0] == 0


def test_requantize_saturates_at_127():
    # 1,000,000 x 678 / 2^16 = 10,345, far above INT8: must clamp to 127, not wrap around
    assert golden.requantize(np.array([1_000_000]), 678, 16)[0] == 127


# ---- quantize.py: every constant fits the hardware ----

def test_weights_fit_int8():
    params, model, X_train, X_test, y_test = build_models()
    for name in ["W1", "W2"]:
        assert np.min(model[name]) >= -127
        assert np.max(model[name]) <= 127


def test_requantize_multipliers_fit_16_bits():
    params, model, X_train, X_test, y_test = build_models()
    assert np.min(model["m1"]) > 0
    assert np.max(model["m1"]) < 2**15


# ---- golden.py: integer model ----

def test_accumulators_fit_24_bits():
    params, model, X_train, X_test, y_test = build_models()
    for images in [X_train, X_test]:
        predictions, acc1, hidden, acc2 = golden.predict(model, images)
        assert golden.bits_needed(acc1) <= golden.ACC_BITS
        assert golden.bits_needed(acc2) <= golden.ACC_BITS


def test_golden_uses_integers_only():
    params, model, X_train, X_test, y_test = build_models()
    predictions, acc1, hidden, acc2 = golden.predict(model, X_test)
    for values in [acc1, hidden, acc2]:
        assert np.issubdtype(values.dtype, np.integer)
    assert np.min(hidden) >= 0
    assert np.max(hidden) <= 127


def test_int8_accuracy_within_one_point_of_float():
    params, model, X_train, X_test, y_test = build_models()
    predictions, acc1, hidden, acc2 = golden.predict(model, X_test)
    int_accuracy = np.mean(predictions == y_test)
    float_accuracy = mlp.accuracy(params, X_test / 16.0, y_test)
    assert float_accuracy - int_accuracy <= 0.01
