import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import ListedColormap

# =========================
# Style (same as Fig. 5)
# =========================
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "mathtext.fontset": "cm"
})

cmap_points = ListedColormap(["#1f77b4", "#d62728"])
cmap_regions = ListedColormap(["#aec7e8", "#ff9896"])

# =========================
# Dataset (blobs)
# =========================
X, y = make_blobs(
    n_samples=90,
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

# Encode labels as {-1, +1}
y_train = 2*y_train - 1
y_test  = 2*y_test  - 1

# Phase encoding in [-1, 1]
scaler = MinMaxScaler(feature_range=(-1, 1))
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# =========================
# Squeezing kernel (Eq. 7–8)
# =========================
def single_mode_overlap(x, xp, c):
    sech2 = 1.0 / (np.cosh(c)**2)
    t2 = np.tanh(c)**2
    return np.sqrt(sech2 / (1 - np.exp(1j*(xp - x)) * t2))

def squeezing_kernel(x, xp, c):
    return np.real(
        single_mode_overlap(x[0], xp[0], c)
        * single_mode_overlap(x[1], xp[1], c)
    )

# Precompute kernel matrices
def kernel_matrix(X1, X2, c):
    K = np.zeros((len(X1), len(X2)))
    for i in range(len(X1)):
        for j in range(len(X2)):
            K[i, j] = squeezing_kernel(X1[i], X2[j], c)
    return K

# =========================
# Kernelized perceptron
# =========================
class KernelPerceptron:
    def __init__(self, kernel, c, epochs=1000):
        self.kernel = kernel
        self.c = c
        self.epochs = epochs

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.alpha = np.zeros(len(X))
        K = kernel_matrix(X, X, self.c)

        for _ in range(self.epochs):
            for i in range(len(X)):
                decision = np.sign(
                    np.sum(self.alpha * self.y * K[:, i])
                )
                if decision == 0:
                    decision = -1
                if decision != self.y[i]:
                    self.alpha[i] += 1

    def predict(self, Xp):
        K = kernel_matrix(self.X, Xp, self.c)
        return np.sign((self.alpha * self.y) @ K)

# =========================
# Plot decision boundary
# =========================
def plot_decision(ax, clf, epoch, c):
    xx, yy = np.meshgrid(
        np.linspace(X_train[:,0].min()-0.3, X_train[:,0].max()+0.3, 300),
        np.linspace(X_train[:,1].min()-0.3, X_train[:,1].max()+0.3, 300)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(grid).reshape(xx.shape)

    ax.contourf(xx, yy, Z, cmap=cmap_regions, alpha=0.6)

    ax.scatter(X_test[:,0], X_test[:,1],
               c=y_test, cmap=cmap_points, s=25)
    ax.scatter(X_train[:,0], X_train[:,1],
               c=y_train, cmap=cmap_points,
               s=50, marker="x")

    train_acc = np.mean(clf.predict(X_train) == y_train)
    test_acc  = np.mean(clf.predict(X_test)  == y_test)

    ax.set_title(
        f"epoch = {epoch}\ntrain={train_acc:.2f}, test={test_acc:.2f}"
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

# =========================
# Run experiment
# =========================
c = 1.5
epochs_list = [1, 50, 500]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
fig.subplots_adjust(top=0.68, wspace=0.2)
fig.suptitle(
    r"Kernelized perceptron with squeezing kernel ($c=1.5$)",
    fontsize=13,
    y=0.93
)

for ax, ep in zip(axes, epochs_list):
    clf = KernelPerceptron(
        kernel=squeezing_kernel,
        c=c,
        epochs=ep
    )
    clf.fit(X_train, y_train)
    plot_decision(ax, clf, ep, c)

plt.show()