"""Train a 64-32-10 float MLP on the digits dataset."""
import numpy as np

from dataset import SEED, load_split
import mlp


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_split()

    # Scale pixels from 0..16 to 0..1 for float training
    X_train = X_train / 16.0
    X_test = X_test / 16.0

    rng = np.random.default_rng(SEED)
    params = mlp.init_params(rng)

    print("accuracy with random weights:", mlp.accuracy(params, X_test, y_test))
    print("loss with random weights:", mlp.loss_fn(params, X_train, y_train))
