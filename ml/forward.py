# numpy forward pass for the 784 -> 128 -> 10 MLP.

import numpy as np

CLASSES = ['boomerang', 'cat', 'eyeglasses', 'face', 'flower',
           'hat', 'house', 'sailboat', 'spider', 'sword']

N_IN = 784      # 28 x 28 pixels
N_HIDDEN = 128  # hidden layer neurons
N_OUT = len(CLASSES)

# replace negatives with 0. Elementwise, so works on any shape.
def relu(v):
    return np.maximum(0, v)

# one image in -> one image out
def forward_single(x, W1, b1, W2, b2):
    """
    x  : (784,)       pixels scaled to 0..1
    W1 : (128, 784)   b1 : (128,)
    W2 : (10, 128)    b2 : (10,)
    returns (predicted class index, logits of shape (10,))
    """
    # Layer 1: each of the 128 rows of W1 dots with x -> 128 numbers.
    # (128, 784) @ (784,) -> (128,)   [784 MACs per neuron, 100,352 total]
    h = relu(W1 @ x + b1)

    # Layer 2: each of the 10 rows of W2 dots with h -> 10 scores.
    # (10, 128) @ (128,) -> (10,)     [128 MACs per class, 1,280 total]
    z = W2 @ h + b2

    # No softmax: it never changes which score is largest, so argmax of the
    # raw scores gives the same prediction. Hardware skips it entirely.
    return int(np.argmax(z)), z

# many drawings in -> measures accuracy
def forward_batch(X, W1, b1, W2, b2):
    """
    X : (B, 784), one drawing per ROW
    returns predictions (B,) and logits (B, 10)

    Why the transpose: images are rows of X, so the dot products we want are
    (row of X) . (row of W1). That's X @ W1.T:
        (B, 784) @ (784, 128) -> (B, 128)
    Each output row is one image's hidden layer. b1 has shape (128,) and numpy
    "broadcasts" it, adding the same bias vector to every row.
    """
    H = relu(X @ W1.T + b1)          # (B, 128)
    Z = H @ W2.T + b2                # (B, 128) @ (128, 10) -> (B, 10)
    return np.argmax(Z, axis=1), Z   # argmax along each row -> (B,)

# function for creating random weights
def random_weights(seed=0):
    """Small random weights just to exercise the shapes."""
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((N_HIDDEN, N_IN)) * 0.01
    b1 = np.zeros(N_HIDDEN)
    W2 = rng.standard_normal((N_OUT, N_HIDDEN)) * 0.01
    b2 = np.zeros(N_OUT)
    return W1, b1, W2, b2

def load_examples(per_class=5, seed=0):
    """Grab a few real drawings from each class. Returns X (B, 784) in 0..1, labels (B,)."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for label, c in enumerate(CLASSES):
        data = np.load(f"data/{c}.npy", mmap_mode='r')
        idx = rng.choice(len(data), per_class, replace=False)
        X.append(data[idx])
        y += [label] * per_class
    X = np.concatenate(X).astype(np.float32) / 255.0   # uint8 0..255 -> float 0..1
    return X, np.array(y)


if __name__ == "__main__":
    W1, b1, W2, b2 = random_weights()
    print("W1", W1.shape, " b1", b1.shape, " W2", W2.shape, " b2", b2.shape)

    X, y = load_examples()
    print("X ", X.shape, " y ", y.shape)

    # Single image, step by step
    x = X[0]
    pred, z = forward_single(x, W1, b1, W2, b2)
    print(f"\nsingle: x {x.shape} -> logits {z.shape} -> pred {pred} ({CLASSES[pred]}), "
          f"true {CLASSES[y[0]]}")

    # Batch
    preds, Z = forward_batch(X, W1, b1, W2, b2)
    print(f"batch:  X {X.shape} -> logits {Z.shape} -> preds {preds.shape}")

    # Sanity check: both versions must agree exactly on every image
    singles = np.array([forward_single(X[i], W1, b1, W2, b2)[0] for i in range(len(X))])
    assert np.array_equal(singles, preds), "single and batch disagree!"
    print("single and batch agree on all", len(X), "images")

    acc = (preds == y).mean()
    print(f"\naccuracy with random weights: {acc:.0%}  (expect ~{1/N_OUT:.0%}, i.e. chance)")