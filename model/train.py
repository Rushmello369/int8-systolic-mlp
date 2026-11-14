"""Train a 64-32-10 float MLP on the digits dataset."""
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from dataset import SEED, load_split
import mlp

# Training hyperparameters
LEARNING_RATE = 0.1   # how big a step each update takes
BATCH_SIZE = 32       # images per update
NUM_EPOCHS = 150      # passes over the training set

# Where the trained weights are saved (model/data/, ignored by git)
WEIGHTS_FILE = Path(__file__).parent / "data" / "float_weights.npz"


def train(params, X_train, y_train, X_val, y_val, rng):
    """Update params in place with mini-batch gradient descent."""
    num_images = len(y_train)

    for epoch in range(1, NUM_EPOCHS + 1):
        # Visit the training images in a new random order every epoch
        order = rng.permutation(num_images)

        for start in range(0, num_images, BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            X_batch = X_train[batch]
            y_batch = y_train[batch]

            grads = mlp.backward(params, X_batch, y_batch)

            # Gradient descent: move every parameter a small step downhill
            for name in params:
                params[name] -= LEARNING_RATE * grads[name]

        # Every 10 epochs, report progress on the whole training and validation sets
        if epoch % 10 == 0:
            train_loss = mlp.loss_fn(params, X_train, y_train)
            train_acc = mlp.accuracy(params, X_train, y_train)
            val_acc = mlp.accuracy(params, X_val, y_val)
            print(f"epoch {epoch:3d}   loss {train_loss:.4f}   "
                  f"train acc {train_acc:.3f}   val acc {val_acc:.3f}")


if __name__ == "__main__":
    X_train_full, X_test, y_train_full, y_test = load_split()

    # Scale pixels from 0..16 to 0..1 for float training
    X_train_full = X_train_full / 16.0
    X_test = X_test / 16.0

    # Hold back 20% of the training images as a validation set for choosing
    # hyperparameters. The test set is used only once, at the very end.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=SEED, stratify=y_train_full)

    rng = np.random.default_rng(SEED)
    params = mlp.init_params(rng)
    print(f"before training: val acc {mlp.accuracy(params, X_val, y_val):.3f}")

    train(params, X_train, y_train, X_val, y_val, rng)

    test_acc = mlp.accuracy(params, X_test, y_test)
    print(f"\nfinal test accuracy: {test_acc:.4f}")

    # Value ranges, needed in step 2 to choose the INT8 scales
    hidden_pre, hidden, logits = mlp.forward(params, X_train_full)
    print(f"max |W1|   = {np.max(np.abs(params['W1'])):.3f}")
    print(f"max |W2|   = {np.max(np.abs(params['W2'])):.3f}")
    print(f"max hidden = {np.max(hidden):.3f}")

    WEIGHTS_FILE.parent.mkdir(exist_ok=True)
    np.savez(WEIGHTS_FILE, W1=params["W1"], b1=params["b1"], W2=params["W2"], b2=params["b2"])
    print(f"saved {WEIGHTS_FILE}")
