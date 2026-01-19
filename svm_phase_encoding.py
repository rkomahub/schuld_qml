import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_moons, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import ListedColormap

cmap_points = ListedColormap(["#1f77b4", "#d62728"])   # blue, red
cmap_regions = ListedColormap(["#aec7e8", "#ff9896"])  # light blue/red

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "mathtext.fontset": "cm"
})

# ---------- Kernel from Eq. (7)-(8) ----------
def single_mode_overlap(x, xp, c):
    """
    Implements Eq. (8) as written:
    <(c,x)|(c,x')> = sqrt( sech^2(c) / (1 - exp(i(x' - x)) * tanh^2(c)) )
    """
    sech2 = 1.0 / (np.cosh(c) ** 2)
    t2 = np.tanh(c) ** 2
    denom = 1.0 - np.exp(1j * (xp - x)) * t2
    return np.sqrt(sech2 / denom)

def kernel_eq7(X, Y, c, mode="abs2"):
    """
    Eq. (7): product over features i of single-mode overlaps.

    mode:
      - "abs2": use |prod|^2 (always real, nonnegative; very stable for SVC)
      - "real": use Re(prod)  (closer to literal 'overlap', but can be less stable)
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    nX, d = X.shape
    nY = Y.shape[0]
    K = np.empty((nX, nY), dtype=np.float64)

    for a in range(nX):
        # vectorize over Y for speed
        prod = np.ones(nY, dtype=np.complex128)
        for i in range(d):
            prod *= single_mode_overlap(X[a, i], Y[:, i], c)
        if mode == "abs2":
            K[a, :] = np.abs(prod) ** 2
        elif mode == "real":
            K[a, :] = np.real(prod)
        else:
            raise ValueError("mode must be 'abs2' or 'real'")
    return K

# ---------- Plotting helper ----------
def plot_decision(ax, Xtr, ytr, Xte, yte, c, title, mode="abs2"):
    # Precompute kernels
    Ktr = kernel_eq7(Xtr, Xtr, c, mode=mode)
    Kte = kernel_eq7(Xte, Xtr, c, mode=mode)

    clf = SVC(kernel="precomputed", C=1.0)
    clf.fit(Ktr, ytr)

    acc_tr = clf.score(Ktr, ytr)
    acc_te = clf.score(Kte, yte)

    # Decision region on a grid (needs K(grid, Xtr))
    x0min, x0max = Xtr[:,0].min()-0.1, Xtr[:,0].max()+0.1
    x1min, x1max = Xtr[:,1].min()-0.1, Xtr[:,1].max()+0.1
    xx, yy = np.meshgrid(np.linspace(x0min, x0max, 250),
                         np.linspace(x1min, x1max, 250))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Kgrid = kernel_eq7(grid, Xtr, c, mode=mode)
    Z = clf.predict(Kgrid).reshape(xx.shape)

    ax.contourf(xx, yy, Z, cmap=cmap_regions, alpha=0.6)

    # test points (filled circles)
    ax.scatter(
        Xte[:,0], Xte[:,1],
        c=yte, cmap=cmap_points,
        s=25, edgecolors="none"
    )

    # training points (crosses)
    ax.scatter(
        Xtr[:,0], Xtr[:,1],
        c=ytr, cmap=cmap_points,
        s=50, marker="x", linewidths=1.2
    )

    ax.set_title(
        f"{title}\ntrain={acc_tr:.2f}, test={acc_te:.2f}",
        fontsize=11
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

# ---------- Data + reproduction (row 1) ----------
def prep_data(generator, n_train, n_test, seed=0):
    X, y = generator()
    # Map each feature to [0, 2pi] so "x" acts like a phase variable (important!)
    scaler = MinMaxScaler(feature_range=(0.0, 2*np.pi))
    X = scaler.fit_transform(X)
    return train_test_split(X, y, train_size=n_train, test_size=n_test, random_state=seed, stratify=y)

datasets = [
    ("circles", lambda: make_circles(n_samples=200, factor=0.5, noise=0.08, random_state=0)),
    ("moons",   lambda: make_moons(n_samples=200, noise=0.12, random_state=0)),
    ("blobs",   lambda: make_blobs(n_samples=200, centers=2, cluster_std=1.6, random_state=0)),
]

fig, axes = plt.subplots(2, 3, figsize=(11, 6))
plt.subplots_adjust(wspace=0.15, hspace=0.35)
c0 = 1.0
mode = "abs2"  # try "real" if you want the literal overlap (may be less robust)

for j, (name, gen) in enumerate(datasets):
    Xtr, Xte, ytr, yte = prep_data(gen, n_train=50, n_test=150, seed=0)
    plot_decision(axes[0, j], Xtr, ytr, Xte, yte, c=c0, title=name, mode=mode)

# ---------- Row 2: varying c on a larger dataset ----------
Xtr, Xte, ytr, yte = prep_data(
    lambda: make_moons(n_samples=600, noise=0.18, random_state=1),
    n_train=500, n_test=100, seed=1
)
cs = [1.0, 1.5, 2.0]
for j, c in enumerate(cs):
    plot_decision(axes[1, j], Xtr, ytr, Xte, yte, c=c, title=rf"moons ($c={c}$)", mode=mode)

plt.tight_layout()
plt.show()