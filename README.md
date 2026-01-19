Absolutely — here is a **single, self-contained `README.md`**, written so that **someone with zero context** can clone the repo and **make the program work on the first try**.

You can copy-paste this **verbatim**.

---

```md
# Fock-Space Quantum Neural Network (Continuous-Variable QML)

This repository implements a **continuous-variable (CV) quantum neural network**
realizing a **Fock-space classifier**, inspired by

> M. Schuld et al., *Quantum machine learning in feature Hilbert space*.

The model embeds classical data into an **infinite-dimensional (truncated) Fock space**
and performs **linear classification in feature space** using a **non-Gaussian CV quantum circuit**.

---

## Requirements (IMPORTANT)

This project **only works with specific pinned versions**.
Do **not** upgrade packages.

Tested on:
- Ubuntu 22.04
- Python 3.10

---

## Repository structure

```bash
schuld_qml/
├── .venv/                 # Python virtual environment (NOT committed)
├── qm_neural_net.py       # Full Fock-space classifier (training + plots)
├── test_00.py             # Backend sanity check (must return 0.0)
├── requirements.txt       # Fully pinned dependency versions
└── README.md
```

---

## Setup instructions (MANDATORY)

### 1. Create a virtual environment inside the repository
```bash
cd schuld_qml
python3 -m venv .venv
source .venv/bin/activate
```

You must see:

```
(.venv)
```

---

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 3. Verify the CV Fock backend (CRITICAL STEP)

Run:

```bash
python test_00.py
```

Expected output:

```text
0.0
```

If and only if this prints `0.0`, the environment is correct.

---

## Running the classifier

```bash
python qm_neural_net.py
```

This will:

* generate a two-moons dataset,
* train a **2-mode Fock-space classifier** using stochastic gradient descent,
* plot:

  * training loss vs optimization steps,
  * decision regions showing ( p(y=1) ).

---

## Model overview

* **Feature map**
  Classical input ( x \in \mathbb{R}^2 ) is encoded via **displacement operators** into a
  truncated **2-mode Fock space**.

* **Feature space**
  Infinite-dimensional bosonic Hilbert space, truncated numerically.

* **Variational circuit**
  Repeated gate blocks consisting of:

  * Beamsplitter (BS)
  * Displacement (D)
  * Quadratic phase (P)
  * Cubic phase (C)

* **Measurement**
  Photon number expectation value.

* **Loss**
  Square loss with gentle ( \ell_2 ) regularization.

The classifier is:

* **linear in Fock space**,
* **nonlinear in input space**,
* **non-Gaussian**, due to cubic phase gates.

---

## Dependency versions (DO NOT CHANGE)

The program is guaranteed to work only with:

```
PennyLane==0.29.1
PennyLane-SF==0.20.1
strawberryfields==0.23.0
thewalrus==0.19.0
autoray==0.6.12
numpy==1.23.5
scipy==1.10.1
```

These versions are pinned in `requirements.txt`.

---

## Common issues

* **`strawberryfields.fock` device not found**
  → Ensure `PennyLane-SF==0.20.1` is installed.

* **SciPy import errors**
  → Ensure `scipy==1.10.1`.

* **Autoray errors**
  → Ensure `autoray==0.6.12`.

If anything breaks, delete `.venv/` and repeat the setup exactly.

---

## Reproducibility rule

> If it runs once, freeze the environment and never upgrade dependencies.

---

## License

Specify if needed (e.g. MIT).

```

---

If you want next, I can:
- annotate `qm_neural_net.py` line-by-line,
- add a **theory section** matching the paper,
- or help you turn this into a **thesis-ready numerical experiment**.

You’re officially past the hard part.
```