# Results

Every number here points to the report or log it came from. No estimates.

| Metric | Value | Source |
| --- | --- | --- |
| Float accuracy (test split, 360 images) | 96.67 % (348 / 360) | `python model/train.py` |
| INT8 accuracy (test split) | – | `model/golden.py` output |

## Float baseline details

Run: `python model/train.py` (seed 0, 150 epochs, learning rate 0.1, batch 32, no L2).

| Quantity | Value | Used for |
| --- | --- | --- |
| Validation accuracy (288 images) | 97.2 % | choosing hyperparameters |
| Training accuracy | 100.0 % | overfitting check |
| max \|W1\| | 1.434 | INT8 weight scale, layer 1 |
| max \|W2\| | 2.237 | INT8 weight scale, layer 2 |
| max hidden activation | 8.660 | INT8 activation scale between layers |
