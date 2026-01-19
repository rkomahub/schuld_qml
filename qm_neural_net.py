# ============================================================
# FOCK-SPACE CLASSIFIER (FIG. 7–8)
# 2-mode CV quantum neural network in truncated Fock space
# ============================================================

# --- Core scientific libraries
import pennylane as qml
from pennylane import numpy as np

# --- Classical ML / plotting
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import strawberryfields as sf


# ============================================================
# GRAPHICS SETTINGS
# ============================================================

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "mathtext.fontset": "cm"
})

cmap_points  = ListedColormap(["#1f77b4", "#d62728"])
cmap_regions = ListedColormap(["#aec7e8", "#ff9896"])


# ============================================================
# HYPERPARAMETERS
# ============================================================

cutoff_dim = 7      # Fock-space truncation (n_max)
modes      = 2       # two-mode CV system
layers     = 2       # number of variational blocks
batch_size = 5
steps      = 3000
lr         = 0.01
l2_reg     = 1e-4


# ============================================================
# DATASET: TWO MOONS
# ============================================================

X, y = make_moons(n_samples=200, noise=0.1)
X = (X - X.mean(axis=0)) / X.std(axis=0)     # rescale for CV stability

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)


# ============================================================
# QUANTUM DEVICE (Fock backend)
# ============================================================

dev = qml.device(
    "strawberryfields.fock",
    wires=modes,
    cutoff_dim=cutoff_dim
)


# ============================================================
# FEATURE MAP  X → |φ(x)⟩  (Squeezing encoding)
# ============================================================

c = 0.6   # squeezing strength (hyperparameter)

def feature_map(x, c):
    """
    Maps classical input x=(x1,x2) into a 2-mode Fock state
    via squeezing operators. This defines the feature space.
    |(c, x1)> ⊗ |(c, x2)>
    """
    # If you want the exact state |(c, x_i)> from the paper expansion,
    # use squeezing magnitude c and squeezing phase x_i:
    qml.Squeezing(c, x[0], wires=0)
    qml.Squeezing(c, x[1], wires=1)


# ============================================================
# VARIATIONAL GATE BLOCK (BS + D + P + C)
# ============================================================

def gate_block(theta):
    """
    One variational block as in Fig. 7:
    - Beamsplitter (entangling)
    - Displacements
    - Quadratic phase (Gaussian)
    - Cubic phase (non-Gaussian)
    """
    bs_theta, bs_phi, d0, d1, p0, p1, c0, c1 = theta

    qml.Beamsplitter(bs_theta, bs_phi, wires=[0, 1])

    qml.Displacement(d0, 0.0, wires=0)
    qml.Displacement(d1, 0.0, wires=1)

    qml.QuadraticPhase(p0, wires=0)
    qml.QuadraticPhase(p1, wires=1)

    qml.CubicPhase(c0, wires=0)
    qml.CubicPhase(c1, wires=1)


# ============================================================
# FULL QUANTUM CLASSIFIER CIRCUIT
# ============================================================

@qml.qnode(dev)
def circuit(theta, x):
    # initial state |0,0> is implicit

    feature_map(x, c)

    # variational circuit W(theta)
    for l in range(layers):
        qml.Beamsplitter(theta[l, 0], theta[l, 1], wires=[0, 1])
        qml.Displacement(theta[l, 2], 0.0, wires=0)
        qml.Displacement(theta[l, 3], 0.0, wires=1)
        qml.QuadraticPhase(theta[l, 4], wires=0)
        qml.QuadraticPhase(theta[l, 5], wires=1)
        qml.CubicPhase(theta[l, 6], wires=0)
        qml.CubicPhase(theta[l, 7], wires=1)

    # full joint Fock distribution p(n1, n2)
    return qml.probs(wires=[0, 1])


# ============================================================
# PROBABILITY MAP AND LOSS FUNCTION
# ============================================================

def prob_y1(theta, x):
    """
    Returns p(y=1) from Fock probabilities:
    p(y=1) = p(0,2) / (p(2,0) + p(0,2))
    """
    probs = circuit(theta, x)

    # IMPORTANT: PennyLane returns probs as a flat vector of length cutoff_dim**modes.
    # Use qml.math.reshape (autograd-safe) before indexing as probs[n1, n2].
    probs = qml.math.reshape(probs, (cutoff_dim, cutoff_dim))

    o0 = probs[2, 0]   # p(2,0)
    o1 = probs[0, 2]   # p(0,2)

    Z = o0 + o1 + 1e-12
    return o1 / Z


def loss(theta, X, y):
    preds = np.array([prob_y1(theta, x) for x in X])
    mse   = np.mean((preds - y)**2)
    reg   = l2_reg * np.sum(theta**2)
    return mse + reg


# ============================================================
# TRAINING LOOP (Stochastic Gradient Descent)
# ============================================================

rng = np.random.default_rng(0)
theta = 0.05 * rng.normal(size=(layers, 8), requires_grad=True)

opt = qml.AdamOptimizer(lr)
loss_history = []

for step in range(steps):
    idx = rng.choice(len(X_train), batch_size, replace=False)
    Xb, yb = X_train[idx], y_train[idx]

    theta, current_loss = opt.step_and_cost(
        lambda t: loss(t, Xb, yb), theta
    )

    loss_history.append(current_loss)

    if step % 100 == 0:
        full_loss = loss(theta, X_train, y_train)
        loss_history.append(full_loss)
    else:
        loss_history.append(np.nan)


    if step % 500 == 0:
        print(f"step {step:5d} | loss = {current_loss:.4f}")

    if step == 2000:
        opt = qml.AdamOptimizer(lr * 0.3)

np.savez(
    "results.npz",
    theta=theta,
    loss_history=loss_history,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test
)