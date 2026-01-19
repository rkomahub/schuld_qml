import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Perceptron
from matplotlib.colors import ListedColormap

# =========================
# Global style (Fig. 5)
# =========================
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "mathtext.fontset": "cm"
})

cmap_points = ListedColormap(["#1f77b4", "#d62728"])
cmap_regions = ListedColormap(["#aec7e8", "#ff9896"])

# =========================
# Dataset (same spirit as Fig. 5)
# =========================
X, y = make_blobs(
    n_samples=90,          # 70 train + 20 test
    centers=2,
    cluster_std=1.6,
    random_state=0
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    train_size=70,
    test_size=20,
    random_state=0,
    stratify=y
)

# Phase encoding → [-1, +1]
scaler = MinMaxScaler(feature_range=(-1, 1))
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# =========================
# Squeezing feature map (real Fock subspace)
# =========================
def squeezed_vacuum_fock_vector(phase, c, n_cut=14):
    """
    Returns coefficients of |(c, phase)> in the even Fock basis |0>,|2>,...,|2*n_cut>.
    (complex vector of length n_cut+1)
    """
    t = np.tanh(c)
    pref = 1.0 / np.sqrt(np.cosh(c))
    v = np.zeros(n_cut + 1, dtype=np.complex128)
    for n in range(n_cut + 1):
        coef = pref * math.sqrt(math.factorial(2*n)) / ( (2**n) * math.factorial(n) )
        v[n] = coef * ((-np.exp(1j * phase) * t) ** n)
    return v

def squeezing_feature_map(X, c, n_cut=14, use_imag=False):
    """
    Maps 2D input x=(x1,x2) to a high-D real feature vector:
      Phi(x) = Re( |(c,x1)> ⊗ |(c,x2)> ) in the truncated even-Fock basis.
    If use_imag=True, concatenate Im part too (often helps numerically).
    """
    Phi = []
    for x1, x2 in X:
        v1 = squeezed_vacuum_fock_vector(x1, c, n_cut=n_cut)
        v2 = squeezed_vacuum_fock_vector(x2, c, n_cut=n_cut)
        psi = np.kron(v1, v2)  # complex vector length (n_cut+1)^2

        if use_imag:
            feat = np.concatenate([psi.real, psi.imag])
        else:
            feat = psi.real  # "perceptron acts on the real subspace"

        Phi.append(feat.astype(np.float64))
    return np.vstack(Phi)

# =========================
# Plot helper (Fig. 5 style)
# =========================
def plot_decision(ax, clf, epoch, c):
    # Grid in input space
    x0_min, x0_max = X_train[:,0].min() - 0.3, X_train[:,0].max() + 0.3
    x1_min, x1_max = X_train[:,1].min() - 0.3, X_train[:,1].max() + 0.3

    xx, yy = np.meshgrid(
        np.linspace(x0_min, x0_max, 300),
        np.linspace(x1_min, x1_max, 300)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Map to Fock space
    Phi_grid = squeezing_feature_map(grid, c)
    Z = clf.predict(Phi_grid).reshape(xx.shape)

    # Plot regions
    ax.contourf(xx, yy, Z, cmap=cmap_regions, alpha=0.6)

    # Test points
    ax.scatter(
        X_test[:,0], X_test[:,1],
        c=y_test, cmap=cmap_points,
        s=25, edgecolors="none"
    )

    # Training points
    ax.scatter(
        X_train[:,0], X_train[:,1],
        c=y_train, cmap=cmap_points,
        s=50, marker="x", linewidths=1.2
    )

    train_acc = clf.score(
        squeezing_feature_map(X_train, c), y_train
    )
    test_acc = clf.score(
        squeezing_feature_map(X_test, c), y_test
    )

    ax.set_title(
        rf"epoch = {epoch}" "\n"
        rf"train={train_acc:.2f}, test={test_acc:.2f}"
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

# =========================
# Main experiment (Fig. 6)
# =========================
c = 1.5
epochs_list = [1, 500, 5000]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for ax, epochs in zip(axes, epochs_list):
    clf = Perceptron(
        max_iter=epochs,
        tol=None,
        penalty=None,       # no regularisation
        random_state=0
    )

    Phi_train = squeezing_feature_map(X_train, c)
    clf.fit(Phi_train, y_train)

    plot_decision(ax, clf, epochs, c)

plt.subplots_adjust(top=0.78, wspace=0.5)

plt.suptitle(
    r"Perceptron decision boundary after squeezing feature map ($c=1.5$)",
    fontsize=13,
    y=0.96
)

plt.show()