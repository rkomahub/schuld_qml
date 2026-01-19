import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D

# ---------- Kernel definitions (Eq. 7–8) ----------

def single_mode_overlap(x, xp, c):
    """
    <(c,x)|(c,x')> = sqrt( sech^2(c) / (1 - exp(i(x' - x)) tanh^2(c)) )
    """
    sech2 = 1.0 / (np.cosh(c)**2)
    t2 = np.tanh(c)**2
    return np.sqrt(sech2 / (1.0 - np.exp(1j * (xp - x)) * t2))


def kappa_sq(xp1, xp2, c):
    """
    kappa_sq((0,0),(xp1,xp2);c)
    """
    return (
        single_mode_overlap(0.0, xp1, c)
        * single_mode_overlap(0.0, xp2, c)
    )

# ---------- Grid ----------
xs = np.linspace(-1, 1, 300)
X1, X2 = np.meshgrid(xs, xs)
cs = [1.0, 1.5, 2.0]

# ---------- Figure with explicit layout ----------
fig = plt.figure(figsize=(14, 4))
gs = gridspec.GridSpec(
    1, 4,
    width_ratios=[1, 1, 1, 0.08],
    wspace=0.25
)

axes = [
    fig.add_subplot(gs[0, i], projection="3d")
    for i in range(3)
]

cax = fig.add_subplot(gs[0, 3])  # colorbar axis

# ---------- Plot ----------
for ax, c in zip(axes, cs):
    K = np.abs(kappa_sq(X1, X2, c))**2

    surf = ax.plot_surface(
        X1, X2, K,
        cmap="viridis",
        linewidth=0,
        antialiased=True
    )

    ax.set_title(rf"$c = {c}$", pad=12)
    ax.set_xlabel(r"$x'_1$")
    ax.set_ylabel(r"$x'_2$")
    ax.set_zlabel(r"$|\kappa_{\mathrm{sq}}|^2$")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.view_init(elev=30, azim=45)

# ---------- Colorbar (isolated, no overlap) ----------
fig.colorbar(surf, cax=cax)

# ---------- Top margin fix ----------
fig.subplots_adjust(top=0.88)

plt.show()