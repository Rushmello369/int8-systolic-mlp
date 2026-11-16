"""load the sklearn digits dataset and split it into train and test sets"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEED = 0

def load_raw():
    """Return all 1797 images as integer pixels (0-16) and their labels (0-9)."""
    digits = load_digits()
    X = digits.data.astype(np.int64)
    y = digits.target.astype(np.int64)
    return X,y

def load_split():
    """Return X_train, X_test, y_train, y_test with an 80/20 stratified split."""
    X, y = load_raw()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X, y = load_raw()
    print("X:", X.shape, X.dtype, "min", X.min(), "max", X.max())
    print("y:", y.shape, "classes", np.unique(y))

    print(f"\nfirst image, label = {y[0]}")
    for row in X[0].reshape(8, 8):
        print(" ".join(f"{v:2d}" for v in row))

    X_train, X_test, y_train, y_test = load_split()
    print("\ntrain:", X_train.shape, "test:", X_test.shape)
    print("test images per class:", np.bincount(y_test))