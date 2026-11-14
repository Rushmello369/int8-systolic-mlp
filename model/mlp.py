"""The 64-32-10 float MLP: parameters, forward pass, loss and gradients."""
import numpy as np

# Network size
NUM_INPUTS = 64 #8x8 pixels per image
NUM_HIDDEN = 32 #hidden neurons
NUM_OUTPUTS = 10 #one score per digit 0-9

def init_params(rng):
    """Create random starting weights and zero biases."""
    W1 = rng.standard_normal((NUM_HIDDEN,NUM_INPUTS)) * np.sqrt(2 / NUM_INPUTS) #generate 32x64 random array with element mostly between -2 & 2, then reduce by a fixed scale factor
    b1 = np.zeros(NUM_HIDDEN) #0 bias
    W2 = rng.standard_normal((NUM_OUTPUTS, NUM_HIDDEN)) * np.sqrt(2 / NUM_HIDDEN) # 10x32 array for second layer
    b2 = np.zeros(NUM_OUTPUTS)

    params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return params

def forward(params, X):
    """Run a batch of images X, shape (N, 64), through the network."""
    hidden_pre = X @ params["W1"].T + params["b1"]      # (N, 32) before ReLU
    hidden = np.maximum(hidden_pre, 0)                  # (N, 32) after ReLU
    logits = hidden @ params["W2"].T + params["b2"]     # (N, 10) one score per digit
    return hidden_pre, hidden, logits


def accuracy(params, X, y):
    """Fraction of images whose highest score is the correct digit."""
    hidden_pre, hidden, logits = forward(params, X)
    predictions = np.argmax(logits, axis=1)    # index of the highest score in each row
    correct = predictions == y                 # True where the guess is right
    return np.mean(correct)

def softmax(logits):
    """Turn each row of scores into probabilities that sum to 1."""
    # Subtract each row's maximum before np.exp so the numbers never get too large.
    # This does not change the result: the same factor cancels in the division below.
    row_max = np.max(logits, axis=1, keepdims=True)
    exps = np.exp(logits - row_max)
    row_sum = np.sum(exps, axis=1, keepdims=True)
    return exps / row_sum

def loss_fn(params, X, y):
    """Average cross-entropy: how wrong the network is on images X with labels y."""
    hidden_pre, hidden, logits = forward(params, X)
    probs = softmax(logits)

    # For each image, pick the probability the network gave to the correct digit
    num_images = len(y)
    prob_of_correct = probs[np.arange(num_images), y]

    losses = -np.log(prob_of_correct)
    return np.mean(losses)

def backward(params, X, y):
    """Return the gradient of loss_fn for every parameter (same shapes as params)."""
    num_images = len(y)
    hidden_pre, hidden, logits = forward(params, X)
    probs = softmax(logits)

    # Step 1: error at the output scores = probabilities - correct answer (one-hot)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(num_images), y] = 1
    d_logits = (probs - one_hot) / num_images      # (N, 10)

    # Step 2: second layer. Gradient = error at the output x value at the input
    d_W2 = d_logits.T @ hidden                     # (10, 32)
    d_b2 = np.sum(d_logits, axis=0)                # (10,)

    # Step 3: send the error back to the hidden layer through the same weights W2
    d_hidden = d_logits @ params["W2"]             # (N, 32)

    # Step 4: back through ReLU. Neurons that were switched off get no gradient
    relu_was_on = hidden_pre > 0
    d_hidden_pre = d_hidden * relu_was_on          # (N, 32)

    # Step 5: first layer, same rule as step 2
    d_W1 = d_hidden_pre.T @ X                      # (32, 64)
    d_b1 = np.sum(d_hidden_pre, axis=0)            # (32,)

    grads = {"W1": d_W1, "b1": d_b1, "W2": d_W2, "b2": d_b2}
    return grads
