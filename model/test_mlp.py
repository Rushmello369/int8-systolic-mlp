"""Checks for the MLP in mlp.py. Run from the project root with: python -m pytest model -v"""
import numpy as np

from dataset import SEED, load_split
import mlp


def load_small_batch():
    """First 50 training images, scaled to 0..1, plus random starting weights."""
    X_train, X_test, y_train, y_test = load_split()
    X = X_train[:50] / 16.0
    y = y_train[:50]
    params = mlp.init_params(np.random.default_rng(SEED))
    return params, X, y


def test_forward_shapes():
    params, X, y = load_small_batch()
    hidden_pre, hidden, logits = mlp.forward(params, X)
    assert hidden_pre.shape == (50, 32)
    assert hidden.shape == (50, 32)
    assert logits.shape == (50, 10)


def test_random_loss_is_near_log10():
    params, X, y = load_small_batch()
    loss = mlp.loss_fn(params, X, y)
    assert 2.0 < loss < 3.0      # random guessing gives about -log(0.1) = 2.3


def test_gradients_match_numerical_estimate():
    """Compare backward() with a slow numerical estimate at one entry of each parameter."""
    params, X, y = load_small_batch()
    grads = mlp.backward(params, X, y)
    eps = 1e-5

    for name in params:
        array = params[name]

        # Check the entry with the largest gradient, so we never compare 0 with 0.
        # array.flat lets us index a 2-D array as if it were 1-D.
        index = np.argmax(np.abs(grads[name]))
        original = array.flat[index]

        array.flat[index] = original + eps
        loss_plus = mlp.loss_fn(params, X, y)

        array.flat[index] = original - eps
        loss_minus = mlp.loss_fn(params, X, y)

        array.flat[index] = original        # put the weight back

        numeric = (loss_plus - loss_minus) / (2 * eps)
        analytic = grads[name].flat[index]
        assert abs(analytic - numeric) < 1e-6, f"{name}[{index}]: backward {analytic} vs numeric {numeric}"
