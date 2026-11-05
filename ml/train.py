"""
train.py — train the 784 -> 128 -> 10 MLP on QuickDraw in PyTorch.

What it does:
  1. Builds a fixed train/test split (cached to data/split.npz so every later
     script — quantization, golden model, test vectors — uses the SAME images).
  2. Trains with Adam + cross-entropy.
  3. Exports W1, b1, W2, b2 as numpy arrays to ml/weights/float_weights.npz.
  4. Re-runs the test set through numpy forward pass (forward.py) and
     checks it agrees with PyTorch. This is the first link in the chain
     PyTorch -> numpy float -> numpy int8 (golden model) -> RTL.
  5. Saves a confusion matrix + training curve to docs/results/.

Run from the repo root:  python ml/train.py
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")            # save plots to files, never open windows
import matplotlib.pyplot as plt

from forward import CLASSES, N_IN, N_HIDDEN, N_OUT, forward_batch

# ---------------- config ----------------
TRAIN_PER_CLASS = 10_000
TEST_PER_CLASS = 2_000
EPOCHS = 15
BATCH_SIZE = 256
LR = 1e-3
SEED = 0

SPLIT_PATH = "data/split.npz"
WEIGHTS_DIR = "ml/weights"
RESULTS_DIR = "docs/results"


# ---------------- data ----------------
def build_split():
    """
    Pick TRAIN+TEST random drawings per class, shuffle, split.
    Cached so the split never changes between runs.
    Stored as uint8 (0..255) — the same format the FPGA will receive.
    """
    if os.path.exists(SPLIT_PATH):
        d = np.load(SPLIT_PATH)
        print(f"loaded cached split from {SPLIT_PATH}")
        return d["X_train"], d["y_train"], d["X_test"], d["y_test"]

    rng = np.random.default_rng(SEED)
    X_train, y_train, X_test, y_test = [], [], [], []
    for label, c in enumerate(CLASSES):
        data = np.load(f"data/{c}.npy", mmap_mode="r")
        idx = rng.choice(len(data), TRAIN_PER_CLASS + TEST_PER_CLASS, replace=False)
        chosen = np.asarray(data[np.sort(idx)])      # sorted reads are faster on mmap
        chosen = chosen[rng.permutation(len(chosen))] # re-shuffle before splitting
        X_train.append(chosen[:TRAIN_PER_CLASS])
        X_test.append(chosen[TRAIN_PER_CLASS:])
        y_train += [label] * TRAIN_PER_CLASS
        y_test += [label] * TEST_PER_CLASS
        print(f"  {c:<12} {len(data):>7} available, using {len(chosen)}")

    X_train, X_test = np.concatenate(X_train), np.concatenate(X_test)
    y_train, y_test = np.array(y_train, dtype=np.int64), np.array(y_test, dtype=np.int64)
    np.savez(SPLIT_PATH, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    print(f"saved split to {SPLIT_PATH}")
    return X_train, y_train, X_test, y_test


def to_float(X_uint8):
    """uint8 0..255 -> float32 0..1, the model's input scale."""
    return X_uint8.astype(np.float32) / 255.0


# ---------------- model ----------------
def build_model():
    # Exactly the network in forward.py:
    #   Linear(784->128): W1 (128,784), b1 (128,)
    #   ReLU
    #   Linear(128->10):  W2 (10,128),  b2 (10,)
    # No softmax layer: CrossEntropyLoss applies it internally during training,
    # and prediction only needs argmax.
    return nn.Sequential(
        nn.Linear(N_IN, N_HIDDEN),
        nn.ReLU(),
        nn.Linear(N_HIDDEN, N_OUT),
    )


@torch.no_grad()
def evaluate(model, X, y):
    model.eval()
    preds = model(X).argmax(dim=1)
    return (preds == y).float().mean().item(), preds.numpy()


def train(model, X_train, y_train, X_test, y_test):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    history = {"loss": [], "test_acc": []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(X_train))   # new shuffle every epoch
        total_loss = 0.0

        for i in range(0, len(perm), BATCH_SIZE):
            b = perm[i:i + BATCH_SIZE]
            logits = model(X_train[b])          # forward pass
            loss = loss_fn(logits, y_train[b])  # how wrong were we?
            opt.zero_grad()
            loss.backward()                     # backprop: compute gradients
            opt.step()                          # nudge weights downhill
            total_loss += loss.item() * len(b)

        train_loss = total_loss / len(perm)
        test_acc, _ = evaluate(model, X_test, y_test)
        history["loss"].append(train_loss)
        history["test_acc"].append(test_acc)
        print(f"epoch {epoch:2d}  loss {train_loss:.4f}  test acc {test_acc:.2%}  ({time.time() - t0:.1f}s)")

    return history


# ---------------- export + checks ----------------
def export_weights(model):
    """Pull weights out of PyTorch as plain numpy arrays, in forward.py's layout."""
    W1 = model[0].weight.detach().numpy().copy()   # (128, 784)
    b1 = model[0].bias.detach().numpy().copy()     # (128,)
    W2 = model[2].weight.detach().numpy().copy()   # (10, 128)
    b2 = model[2].bias.detach().numpy().copy()     # (10,)

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    path = os.path.join(WEIGHTS_DIR, "float_weights.npz")
    np.savez(path, W1=W1, b1=b1, W2=W2, b2=b2)
    print(f"\nsaved weights to {path}")
    for name, a in [("W1", W1), ("b1", b1), ("W2", W2), ("b2", b2)]:
        print(f"  {name} {str(a.shape):<11} range [{a.min():+.3f}, {a.max():+.3f}]")
    return W1, b1, W2, b2


def check_numpy_matches(W1, b1, W2, b2, X_test_f, torch_preds):
    np_preds, _ = forward_batch(X_test_f, W1, b1, W2, b2)
    agree = (np_preds == torch_preds).mean()
    print(f"\nnumpy forward pass agrees with PyTorch on {agree:.2%} of test images")
    # Float rounding can differ in the last bit between libraries, which can flip
    # a near-tie. Anything below ~99.9% means a real bug (transpose, wrong layer...).
    if agree < 0.999:
        print("  WARNING: mismatch too large — check weight layout/transposes")
    return np_preds


def confusion_matrix(y_true, y_pred):
    cm = np.zeros((N_OUT, N_OUT), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)   # cm[true, pred] += 1 for every image
    return cm


def report(cm, history):
    per_class = cm.diagonal() / cm.sum(axis=1)
    print("\nper-class accuracy:")
    for c, a in sorted(zip(CLASSES, per_class), key=lambda t: t[1]):
        print(f"  {c:<12} {a:.1%}")

    off = cm.copy()
    np.fill_diagonal(off, 0)
    print("\ntop confusions (true -> predicted):")
    for flat in np.argsort(off, axis=None)[::-1][:5]:
        t, p = divmod(flat, N_OUT)
        print(f"  {CLASSES[t]:<12} -> {CLASSES[p]:<12} {off[t, p]} images")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # confusion matrix (row-normalized so each row sums to 100%)
    cm_pct = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(N_OUT), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(N_OUT), CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for t in range(N_OUT):
        for p in range(N_OUT):
            v = cm_pct[t, p]
            if v >= 0.01:
                ax.text(p, t, f"{v:.0%}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.5 else "black")
    ax.set_title("Float model — confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "confusion_float.png"), dpi=150)
    plt.close(fig)

    # training curve
    fig, ax1 = plt.subplots(figsize=(7, 4))
    epochs = range(1, len(history["loss"]) + 1)
    ax1.plot(epochs, history["loss"], "o-", label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss")
    ax2 = ax1.twinx()
    ax2.plot(epochs, history["test_acc"], "s-", color="tab:orange", label="test acc")
    ax2.set_ylabel("test accuracy")
    fig.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "training_curve.png"), dpi=150)
    plt.close(fig)
    print(f"\nsaved plots to {RESULTS_DIR}/")


# ---------------- main ----------------
if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    X_train, y_train, X_test, y_test = build_split()
    print(f"train {X_train.shape}, test {X_test.shape}\n")

    X_train_f, X_test_f = to_float(X_train), to_float(X_test)
    Xtr, ytr = torch.from_numpy(X_train_f), torch.from_numpy(y_train)
    Xte, yte = torch.from_numpy(X_test_f), torch.from_numpy(y_test)

    model = build_model()
    history = train(model, Xtr, ytr, Xte, yte)

    final_acc, torch_preds = evaluate(model, Xte, yte)
    print(f"\nfinal test accuracy: {final_acc:.2%}")

    W1, b1, W2, b2 = export_weights(model)
    check_numpy_matches(W1, b1, W2, b2, X_test_f, torch_preds)

    report(confusion_matrix(y_test, torch_preds), history)