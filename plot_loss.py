import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# LOSS CURVE (FIG. 8 LEFT PANEL)
# ============================================================

data = np.load("results.npz", allow_pickle=True)
loss_history = data["loss_history"]

plt.figure(figsize=(4, 3))
plt.plot(loss_history)
plt.xlabel("training step")
plt.ylabel("training loss (full dataset)")
plt.tight_layout()
plt.show()