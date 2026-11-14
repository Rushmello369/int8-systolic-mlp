# Results

Every number here points to the report or log it came from. No estimates.

| Metric | Value | Source |
| --- | --- | --- |
| Float accuracy (test split, 360 images) | 96.67 % (348 / 360) | `python model/train.py` |
| INT8 accuracy (test split, 360 images) | 96.11 % (346 / 360) | `python model/golden.py` |
| INT8 loss vs float | −0.56 points (target ≤ 1) | `python model/golden.py` |

## Float baseline details

Run: `python model/train.py` (seed 0, 150 epochs, learning rate 0.1, batch 32, no L2).

| Quantity | Value | Used for |
| --- | --- | --- |
| Validation accuracy (288 images) | 97.2 % | choosing hyperparameters |
| Training accuracy | 100.0 % | overfitting check |
| max \|W1\| | 1.434 | INT8 weight scale, layer 1 |
| max \|W2\| | 2.237 | INT8 weight scale, layer 2 |
| max hidden activation | 8.660 | INT8 activation scale between layers |

## INT8 model details

Run: `python model/quantize.py`, then `python model/golden.py`. Calibrated on the training split only.

| Quantity | Value | Note |
| --- | --- | --- |
| Images where INT8 and float predict differently | 2 / 360 | |
| Layer 1 weight scale (per neuron) | 0.00272 .. 0.01129 | one scale per row of W1 |
| Hidden activation scale | 0.06819 | 8.660 / 127 |
| Layer 2 weight scale (whole matrix) | 0.01762 | one scale so integer scores compare directly |
| Requantisation multiplier m1 | 164 .. 678, shift 16 | fits 16 bits |
| max \|b1_int\|, max \|b2_int\| | 946, 224 | |
| Largest layer 1 accumulator (test set) | 15,685 (15 bits signed) | worst-case bound 2^20 (21 bits) |
| Largest layer 2 accumulator (test set) | 19,694 (16 bits signed) | |
| Accumulator width chosen | 24 bits | `test_accumulators_fit_24_bits` |
| Hidden INT8 values seen | 0 .. 121 | |
| Python tests | 11 passed | `python -m pytest model -v` |
