import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pennylane as qml

# --- load results
data = np.load("results.npz", allow_pickle=True)
theta   = data["theta"]
X_train = data["X_train"]
y_train = data["y_train"]
X_test  = data["X_test"]
y_test  = data["y_test"]

# --- colormaps
cmap_points  = ListedColormap(["#1f77b4", "#d62728"])

# --- prob_y1 and circuit must be imported or redefined
from qm_neural_net import prob_y1

# ============================================================
# DECISION REGIONS (FIG. 8 RIGHT PANEL)
# ============================================================

# --- accuracy
def accuracy(theta, X, y):
    preds = np.array([int(prob_y1(theta, x) > 0.5) for x in X])
    return np.mean(preds == y)

acc_train = accuracy(theta, X_train, y_train)
acc_test  = accuracy(theta, X_test,  y_test)

# --- grid
xx, yy = np.meshgrid(
    np.linspace(X_train[:,0].min()-0.5, X_train[:,0].max()+0.5, 200),
    np.linspace(X_train[:,1].min()-0.5, X_train[:,1].max()+0.5, 200)
)

grid = np.c_[xx.ravel(), yy.ravel()]
Z = np.array([prob_y1(theta, x) for x in grid])
Z = Z.reshape(xx.shape)

# --- plot
plt.figure(figsize=(6, 5))
cont = plt.contourf(xx, yy, Z, levels=50, cmap="coolwarm")
plt.colorbar(cont, label=r"$p(y=1)$")

# Training points (crosses)
plt.scatter(X_train[:,0], X_train[:,1],
            c=y_train, cmap=cmap_points,
            marker="x", s=40, label="Train")

# Test points (circles)
plt.scatter(X_test[:,0], X_test[:,1],
            c=y_test, cmap=cmap_points,
            edgecolors="k", s=40, label="Test")

plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.title("Fock-space classifier")
plt.suptitle(
    rf"Train acc = {acc_train:.2f}, Test acc = {acc_test:.2f}",
    fontsize=10
)
plt.legend()
plt.tight_layout()
plt.show()