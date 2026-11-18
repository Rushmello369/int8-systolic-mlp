"""Write the memory files the hardware loads, and the results it must reproduce.

Reads  model/data/int8_model.npz  (written by quantize.py)
Writes model/data/*.mem           (read by the RTL with $readmemh, and by the testbenches)

Every .mem file has one memory word per line, in hexadecimal. A word holds 8 lanes
(one per row or column of the 8x8 array). Lane 0 sits in the lowest bits, so in the
written hex string lane 0 is the rightmost group of digits.

Memory layouts (they must match rtl/core/mlp_core.sv):
  weights.mem   one word per weight-load cycle: tile t, cycle c is line t*8 + c.
                Lane n = W[neuron n of the tile, input (7 - c) of the tile]:
                the bottom row is sent first because it is pushed furthest down.
                Tiles 0..31 are layer 1 (tile = column_block*8 + row_block),
                tiles 32..39 are layer 2 (tile = 32 + column_block*4 + row_block).
  images.mem    image i, row block k is line i*8 + k; lane j = pixel k*8 + j.
  labels.mem    one label per image.
  bias.mem      blocks 0..3: b1 for neurons 8b..8b+7; blocks 4..5: b2, padded with 0.
  mult.mem      blocks 0..3: m1 for neurons 8b..8b+7.
Expected results:
  expected_hidden.mem   image i, block b is line i*4 + b; lane n = hidden value 8b+n.
  expected_scores.mem   image i, block b is line i*2 + b; lane n = score of digit 8b+n
                        (lanes past digit 9 are 0).
  expected_pred.mem     one predicted digit per image.
  expected_summary.mem  line 0: number of correct predictions; line 1: checksum,
                        the sum of all 10 scores of every image, modulo 2^32.
"""
from pathlib import Path

import numpy as np

from dataset import load_split
import golden

DATA_DIR = Path(__file__).parent / "data"

ROWS = 8            # array rows: inputs per tile
COLS = 8            # array columns: neurons per tile
DATA_BITS = 8       # weights, pixels and hidden values
ACC_BITS = 24       # accumulators and biases
MULT_BITS = 16      # requantisation multipliers m1
NUM_IMAGES = 360    # test split size; rtl/pkg/accel_pkg.sv has the same number
PADDED_OUTPUTS = 16  # layer 2 has 10 neurons, padded to two full column blocks


def pack_lanes(values, lane_bits):
    """Pack a list of integers into one word, lane 0 in the lowest bits.

    Negative numbers are stored in two's complement, lane_bits wide.
    Example: pack_lanes([1, -1], 8) = 0xFF01.
    """
    mask = (1 << lane_bits) - 1
    word = 0
    for lane in range(len(values)):
        lane_value = int(values[lane]) & mask       # two's complement in lane_bits
        word = word | (lane_value << (lane * lane_bits))
    return word


def write_mem(file_name, words, word_bits):
    """Write one hex word per line, zero-padded to the full word width."""
    hex_digits = (word_bits + 3) // 4
    lines = []
    for word in words:
        lines.append(format(word, "0" + str(hex_digits) + "x"))
    path = DATA_DIR / file_name
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path.name}: {len(lines)} words of {word_bits} bits")


def weight_tile_words(weights, column_block, row_block):
    """The 8 load words of one tile, bottom row first.

    weights is a (neurons, inputs) matrix. The tile covers neurons
    column_block*8 .. +7 and inputs row_block*8 .. +7.
    """
    words = []
    for cycle in range(ROWS):
        row = ROWS - 1 - cycle                     # bottom row is loaded first
        input_index = row_block * ROWS + row
        lanes = []
        for col in range(COLS):
            neuron = column_block * COLS + col
            lanes.append(weights[neuron, input_index])
        words.append(pack_lanes(lanes, DATA_BITS))
    return words


def main():
    model = golden.load_int_model()
    X_train, X_test, y_train, y_test = load_split()
    assert len(X_test) == NUM_IMAGES, "test split size changed: update accel_pkg.sv too"

    W1 = model["W1"]                 # (32, 64)
    W2 = model["W2"]                 # (10, 32)
    num_hidden = W1.shape[0]
    num_outputs = W2.shape[0]

    # Layer 2 padded with zero rows to 16 neurons, so it fills two column blocks
    W2_padded = np.zeros((PADDED_OUTPUTS, W2.shape[1]), dtype=np.int64)
    W2_padded[:num_outputs, :] = W2
    b2_padded = np.zeros(PADDED_OUTPUTS, dtype=np.int64)
    b2_padded[:num_outputs] = model["b2"]

    # ---- Weights: layer 1 tiles, then layer 2 tiles ----
    weight_words = []
    for column_block in range(num_hidden // COLS):            # 4
        for row_block in range(W1.shape[1] // ROWS):         # 8
            weight_words += weight_tile_words(W1, column_block, row_block)
    for column_block in range(PADDED_OUTPUTS // COLS):       # 2
        for row_block in range(W2.shape[1] // ROWS):         # 4
            weight_words += weight_tile_words(W2_padded, column_block, row_block)
    write_mem("weights.mem", weight_words, COLS * DATA_BITS)

    # ---- Images and labels ----
    image_words = []
    for image in range(NUM_IMAGES):
        for row_block in range(X_test.shape[1] // ROWS):
            pixels = X_test[image, row_block * ROWS:(row_block + 1) * ROWS]
            image_words.append(pack_lanes(pixels, DATA_BITS))
    write_mem("images.mem", image_words, ROWS * DATA_BITS)
    write_mem("labels.mem", [int(label) for label in y_test], 4)

    # ---- Biases and requantisation multipliers, one word per column block ----
    bias_words = []
    for block in range(num_hidden // COLS):
        bias_words.append(pack_lanes(model["b1"][block * COLS:(block + 1) * COLS], ACC_BITS))
    for block in range(PADDED_OUTPUTS // COLS):
        bias_words.append(pack_lanes(b2_padded[block * COLS:(block + 1) * COLS], ACC_BITS))
    write_mem("bias.mem", bias_words, COLS * ACC_BITS)

    mult_words = []
    for block in range(num_hidden // COLS):
        mult_words.append(pack_lanes(model["m1"][block * COLS:(block + 1) * COLS], MULT_BITS))
    write_mem("mult.mem", mult_words, COLS * MULT_BITS)

    # ---- Expected results from the integer golden model ----
    predictions, acc1, hidden, acc2 = golden.predict(model, X_test)

    hidden_words = []
    for image in range(NUM_IMAGES):
        for block in range(num_hidden // COLS):
            hidden_words.append(pack_lanes(hidden[image, block * COLS:(block + 1) * COLS], DATA_BITS))
    write_mem("expected_hidden.mem", hidden_words, COLS * DATA_BITS)

    score_words = []
    for image in range(NUM_IMAGES):
        padded_scores = np.zeros(PADDED_OUTPUTS, dtype=np.int64)
        padded_scores[:num_outputs] = acc2[image]
        for block in range(PADDED_OUTPUTS // COLS):
            score_words.append(pack_lanes(padded_scores[block * COLS:(block + 1) * COLS], ACC_BITS))
    write_mem("expected_scores.mem", score_words, COLS * ACC_BITS)

    write_mem("expected_pred.mem", [int(digit) for digit in predictions], 4)

    num_correct = int(np.sum(predictions == y_test))
    checksum = int(np.sum(acc2)) % (1 << 32)
    write_mem("expected_summary.mem", [num_correct, checksum], 32)
    print(f"correct: {num_correct} / {NUM_IMAGES}, checksum: 0x{checksum:08x}")


if __name__ == "__main__":
    main()
