"""Integer-only reference model. The hardware must match these outputs bit for bit.

Reads model/data/int8_model.npz (written by quantize.py). No floating-point arithmetic
is used anywhere in the inference path below.
"""
from pathlib import Path

import numpy as np

from dataset import load_split

INT_MODEL_FILE = Path(__file__).parent / "data" / "int8_model.npz"
ACC_BITS = 24   # signed accumulator width in the hardware


def load_int_model():
    """Read the integer constants saved by quantize.py."""
    data = np.load(INT_MODEL_FILE)
    model = {}
    for name in ["W1", "b1", "m1", "W2", "b2"]:
        model[name] = data[name].astype(np.int64)
    model["shift"] = int(data["shift"])
    return model


def requantize(acc, m, shift):
    """Change an accumulator to the INT8 activation unit, then apply ReLU.

    1. acc x m                 integer multiply
    2. + 2^(shift-1)           add half a step so the shift rounds to nearest
    3. >> shift                arithmetic right shift (rounds down, like >>> in SystemVerilog)
    4. clamp to 0..127         saturate to INT8; the lower bound 0 is the ReLU
    """
    scaled = acc * m
    half_step = 1 << (shift - 1)
    shifted = (scaled + half_step) >> shift
    return np.clip(shifted, 0, 127)


def layer1(model, pixels):
    """pixels: (N, 64) integers 0..16. Returns the accumulators (N, 32) and INT8 hidden values (N, 32)."""
    acc1 = pixels @ model["W1"].T + model["b1"]            # integer multiply-accumulate, plus bias
    hidden = requantize(acc1, model["m1"], model["shift"])  # each neuron uses its own m
    return acc1, hidden


def layer2(model, hidden):
    """hidden: (N, 32) integers 0..127. Returns one integer score per digit, shape (N, 10)."""
    acc2 = hidden @ model["W2"].T + model["b2"]
    return acc2


def predict(model, pixels):
    """Return the predicted digit for each image, plus every intermediate value."""
    acc1, hidden = layer1(model, pixels)
    acc2 = layer2(model, hidden)

    # Largest score wins. On a tie np.argmax picks the lowest digit; the hardware must do the same.
    predictions = np.argmax(acc2, axis=1)
    return predictions, acc1, hidden, acc2


def bits_needed(values):
    """Signed bit width needed to hold every value in the array."""
    largest = int(np.max(np.abs(values)))
    return largest.bit_length() + 1     # +1 for the sign bit


if __name__ == "__main__":
    import mlp
    from quantize import load_float_params

    model = load_int_model()
    X_train, X_test, y_train, y_test = load_split()

    predictions, acc1, hidden, acc2 = predict(model, X_test)
    int_accuracy = np.mean(predictions == y_test)

    # Float accuracy on the same 360 images, for comparison only
    float_params = load_float_params()
    float_accuracy = mlp.accuracy(float_params, X_test / 16.0, y_test)
    hidden_pre, float_hidden, float_logits = mlp.forward(float_params, X_test / 16.0)
    float_predictions = np.argmax(float_logits, axis=1)
    num_disagree = np.sum(predictions != float_predictions)

    print(f"float test accuracy: {float_accuracy:.4f}  ({np.sum(float_predictions == y_test)}/{len(y_test)})")
    print(f"INT8  test accuracy: {int_accuracy:.4f}  ({np.sum(predictions == y_test)}/{len(y_test)})")
    print(f"difference:          {100 * (float_accuracy - int_accuracy):+.2f} points")
    print(f"images where INT8 and float predict differently: {num_disagree}")

    print(f"\nlayer 1 accumulator: max |acc1| = {np.max(np.abs(acc1))}, needs {bits_needed(acc1)} bits")
    print(f"layer 2 accumulator: max |acc2| = {np.max(np.abs(acc2))}, needs {bits_needed(acc2)} bits")
    print(f"accumulator width in hardware: {ACC_BITS} bits")
    print(f"hidden values: {np.min(hidden)} .. {np.max(hidden)}")
