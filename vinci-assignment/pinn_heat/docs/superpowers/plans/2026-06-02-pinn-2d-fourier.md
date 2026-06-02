# PINN 2D Fourier Feature Embeddings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `pinn_2d_fourier.ipynb` — a Fourier-feature PINN for the 2D heat equation with a multi-frequency IC — and compare its accuracy against a plain FNN baseline trained on the same problem.

**Architecture:** `MultiscaleFourierNet` encodes `(x,y,t)` through three frozen random Fourier bands (σ=1,5,20, m=64 each → 384 features), then passes through a 3-layer MLP (128 neurons, tanh). A plain FNN (5×32, tanh) is also trained on the same IC for direct comparison.

**Tech Stack:** Python, DeepXDE 1.15, PyTorch, NumPy, Matplotlib

---

## File Map

| File | Role |
|---|---|
| `pinn_2d_fourier.ipynb` | New notebook — all cells for Fourier PINN + baseline comparison |

---

### Task 1: Create notebook and install dependencies

**Files:**
- Create: `pinn_2d_fourier.ipynb`

- [ ] **Step 1: Create the notebook with install cell**

```bash
cd /Users/anjali/Projects/physics_deeplearning/vinci-assignment/pinn_heat
python3.11 -c "
import nbformat
nb = nbformat.v4.new_notebook()
nb.cells = [nbformat.v4.new_code_cell('!pip install deepxde torch numpy matplotlib')]
with open('pinn_2d_fourier.ipynb', 'w') as f:
    nbformat.write(nb, f)
print('Created pinn_2d_fourier.ipynb')
"
```

- [ ] **Step 2: Verify**

```bash
ls -la pinn_2d_fourier.ipynb
```
Expected: file exists, size > 0.

- [ ] **Step 3: Commit**

```bash
git add pinn_2d_fourier.ipynb
git commit -m "feat: create pinn_2d_fourier notebook"
```

---

### Task 2: Imports and config

**Files:**
- Modify: `pinn_2d_fourier.ipynb` (add cell)

- [ ] **Step 1: Add imports and config cell**

```python
import os
os.environ["DDE_BACKEND"] = "pytorch"

import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# --- Config ---
ALPHA      = 1.0
T_MAX      = 0.01
N_DOMAIN   = 10000
N_BOUNDARY = 800
N_INITIAL  = 500
W_PDE = 1
W_BC  = 10
W_IC  = 10
T_EVAL = [0.001, 0.005, 0.010]

# Fourier feature config
SIGMA_BANDS = [1, 5, 20]   # frequency band scales
M_FREQ      = 64           # random frequencies per band
# Total Fourier features: len(SIGMA_BANDS) * 2 * M_FREQ = 384

print(f"Backend: {dde.backend.backend_name}")
print(f"T_MAX={T_MAX}")
print(f"Fourier features: {len(SIGMA_BANDS)} bands × 2 × {M_FREQ} = {len(SIGMA_BANDS)*2*M_FREQ}")
print("Config loaded.")
```

- [ ] **Step 2: Run cell, verify**

Output contains:
```
Backend: pytorch
T_MAX=0.01
Fourier features: 3 bands × 2 × 64 = 384
Config loaded.
```

- [ ] **Step 3: Commit**

```bash
git add pinn_2d_fourier.ipynb
git commit -m "feat: add imports and config"
```

---

### Task 3: Multi-frequency IC and analytical solution

**Files:**
- Modify: `pinn_2d_fourier.ipynb` (add cell)

- [ ] **Step 1: Add IC and analytical solution cell**

```python
def T_exact(x, y, t):
    """
    Two-mode analytical solution:
      mode 1 (low freq):  sin(πx)sin(πy) · exp(−2π²t)
      mode 2 (high freq): 0.3·sin(5πx)sin(5πy) · exp(−50π²t)
    """
    low  = np.sin(np.pi * x) * np.sin(np.pi * y) * np.exp(-2  * np.pi**2 * ALPHA * t)
    high = 0.3 * np.sin(5 * np.pi * x) * np.sin(5 * np.pi * y) * np.exp(-50 * np.pi**2 * ALPHA * t)
    return low + high

# Sanity checks
# 1. At t=0, T_exact should equal the IC
val = T_exact(np.array([0.5]), np.array([0.1]), 0.0)
ic  = np.sin(np.pi*0.5)*np.sin(np.pi*0.1) + 0.3*np.sin(5*np.pi*0.5)*np.sin(5*np.pi*0.1)
assert np.isclose(val[0], ic), f"IC mismatch: {val[0]} vs {ic}"
print(f"IC sanity check passed: T_exact(0.5, 0.1, 0) = {val[0]:.6f}")

# 2. Show high-freq mode amplitude at eval times
print("\nHigh-freq mode amplitude at eval times:")
for t in T_EVAL:
    amp = 0.3 * np.exp(-50 * np.pi**2 * ALPHA * t)
    print(f"  t={t:.3f}: {amp:.4f}")
```

- [ ] **Step 2: Run cell, verify**

Output:
```
IC sanity check passed: T_exact(0.5, 0.1, 0) = <some value>

High-freq mode amplitude at eval times:
  t=0.001: 0.18xx
  t=0.005: 0.02xx
  t=0.010: 0.00xx
```

- [ ] **Step 3: Commit**

```bash
git add pinn_2d_fourier.ipynb
git commit -m "feat: add multi-freq IC and analytical solution"
```

---

### Task 4: Domain definition

**Files:**
- Modify: `pinn_2d_fourier.ipynb` (add cell)

- [ ] **Step 1: Add domain cell**

```python
geom       = dde.geometry.Rectangle(xmin=[0, 0], xmax=[1, 1])
timedomain = dde.geometry.TimeDomain(0, T_MAX)
geomtime   = dde.geometry.GeometryXTime(geom, timedomain)

print(f"Domain: [0,1]² × [0, {T_MAX}]")
print("GeometryXTime created successfully.")
```

- [ ] **Step 2: Run cell, verify**

Prints both lines without error.

- [ ] **Step 3: Commit**

```bash
git add pinn_2d_fourier.ipynb
git commit -m "feat: define space-time domain"
```

---

### Task 5: PDE residual function

**Files:**
- Modify: `pinn_2d_fourier.ipynb` (add cell)

- [ ] **Step 1: Add PDE cell**

```python
def pde(x, u):
    """
    Residual: ∂u/∂t − α(∂²u/∂x² + ∂²u/∂y²) = 0
    x[:,0]=x_coord, x[:,1]=y_coord, x[:,2]=t
    """
    u_t  = dde.grad.jacobian(u, x, i=0, j=2)
    u_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
    u_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)
    return u_t - ALPHA * (u_xx + u_yy)

print("PDE residual function defined.")
```

- [ ] **Step 2: Run cell, verify**

Prints: `PDE residual function defined.`

- [ ] **Step 3: Commit**

```bash
git add pinn_2d_fourier.ipynb
git commit -m "feat: define PDE residual"
```

---

### Task 6: BC and IC definitions

**Files:**
- Modify: `pinn_2d_fourier.ipynb` (add cell)

- [ ] **Step 1: Add BC and IC cell**

```python
# Dirichlet BC: T = 0 on all boundaries
bc = dde.icbc.DirichletBC(
    geomtime,
    func=lambda x: np.zeros((len(x), 1)),
    on_boundary=lambda x, on_boundary: on_boundary,
)

# IC: T(x,y,0) = sin(πx)sin(πy) + 0.3·sin(5πx)sin(5πy)
def ic_func(x):
    low  = np.sin(np.pi * x[:, 0:1]) * np.sin(np.pi * x[:, 1:2])
    high = 0.3 * np.sin(5 * np.pi * x[:, 0:1]) * np.sin(5 * np.pi * x[:, 1:2])
    return low + high

ic = dde.icbc.IC(
    geomtime,
    func=ic_func,
    on_initial=lambda x, on_initial: on_initial,
)

print("BC (Dirichlet, T=0) defined.")
print("IC (two-mode sinusoidal) defined.")
```

- [ ] **Step 2: Run cell, verify**

Prints both lines without error.

- [ ] **Step 3: Commit**

```bash
git add pinn_2d_fourier.ipynb
git commit -m "feat: define BC and multi-freq IC"
```

---

### Task 7: Dataset assembly

**Files:**
- Modify: `pinn_2d_fourier.ipynb` (add cell)

- [ ] **Step 1: Add dataset cell**

```python
def solution_func(x):
    """Analytical solution for DeepXDE test metric."""
    low  = (np.sin(np.pi * x[:, 0:1]) * np.sin(np.pi * x[:, 1:2])
            * np.exp(-2 * np.pi**2 * ALPHA * x[:, 2:3]))
    high = (0.3 * np.sin(5 * np.pi * x[:, 0:1]) * np.sin(5 * np.pi * x[:, 1:2])
            * np.exp(-50 * np.pi**2 * ALPHA * x[:, 2:3]))
    return low + high

data = dde.data.TimePDE(
    geometryxtime=geomtime,
    pde=pde,
    ic_bcs=[bc, ic],
    num_domain=N_DOMAIN,
    num_boundary=N_BOUNDARY,
    num_initial=N_INITIAL,
    train_distribution="LHS",
    solution=solution_func,
    num_test=1000,
)

print("Dataset assembled:")
print(f"  Domain : {N_DOMAIN} pts (LHS)")
print(f"  Boundary: {N_BOUNDARY} pts")
print(f"  IC      : {N_INITIAL} pts")
print(f"  Test    : 1000 pts vs analytical solution")
```

- [ ] **Step 2: Run cell, verify**

Prints dataset summary without error.

- [ ] **Step 3: Commit**

```bash
git add pinn_2d_fourier.ipynb
git commit -m "feat: assemble TimePDE dataset"
```

---

### Task 8: MultiscaleFourierNet class

**Files:**
- Modify: `pinn_2d_fourier.ipynb` (add cell)

- [ ] **Step 1: Add network class cell**

```python
class MultiscaleFourierNet(dde.nn.NN):
    """
    Fourier feature PINN network.
    Encodes (x,y,t) through multiple Fourier bands before an MLP.
    """
    def __init__(self, sigma_bands, m_freq, mlp_hidden, activation="tanh"):
        super().__init__()
        self.sigma_bands = sigma_bands
        n_fourier = len(sigma_bands) * 2 * m_freq  # 3 * 2 * 64 = 384

        # Frozen random frequency matrices, one per band
        for i, sigma in enumerate(sigma_bands):
            B = torch.randn(3, m_freq) * sigma   # shape (3, m_freq)
            self.register_parameter(
                f"B_{i}", nn.Parameter(B, requires_grad=False)
            )

        # MLP layers: n_fourier → hidden → ... → 1
        sizes = [n_fourier] + mlp_hidden + [1]
        layers = []
        act    = nn.Tanh() if activation == "tanh" else nn.ReLU()
        for in_sz, out_sz in zip(sizes[:-2], sizes[1:-1]):
            layers += [nn.Linear(in_sz, out_sz), act]
        layers.append(nn.Linear(sizes[-2], sizes[-1]))  # output: linear
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (N, 3)  columns: [x_coord, y_coord, t]
        features = []
        for i in range(len(self.sigma_bands)):
            B = getattr(self, f"B_{i}")        # (3, m_freq)
            proj = x @ B                       # (N, m_freq)
            features.append(torch.sin(proj))
            features.append(torch.cos(proj))
        z = torch.cat(features, dim=-1)        # (N, 384)
        return self.mlp(z)

    def output_transform(self, x, y):
        return y

# Quick shape test
_net_test = MultiscaleFourierNet(SIGMA_BANDS, M_FREQ, [128, 128, 128])
_x_test   = torch.randn(10, 3)
_out      = _net_test(_x_test)
assert _out.shape == (10, 1), f"Expected (10,1), got {_out.shape}"
print(f"MultiscaleFourierNet shape test passed: (10,3) → {_out.shape}")

total_params = sum(p.numel() for p in _net_test.parameters())
trainable    = sum(p.numel() for p in _net_test.parameters() if p.requires_grad)
print(f"Total parameters : {total_params:,}")
print(f"Trainable params : {trainable:,}  (frozen Fourier matrices excluded)")
del _net_test, _x_test, _out
```

- [ ] **Step 2: Run cell, verify**

Output:
```
MultiscaleFourierNet shape test passed: (10,3) → torch.Size([10, 1])
Total parameters : <N>
Trainable params : <M>  (frozen Fourier matrices excluded)
```
Trainable params should be less than total (frozen B matrices not counted).

- [ ] **Step 3: Commit**

```bash
git add pinn_2d_fourier.ipynb
git commit -m "feat: implement MultiscaleFourierNet with shape test"
```

---

### Task 9: Fourier model construction and training

**Files:**
- Modify: `pinn_2d_fourier.ipynb` (add cell)

- [ ] **Step 1: Add Fourier model training cell**

```python
fourier_net   = MultiscaleFourierNet(SIGMA_BANDS, M_FREQ, [128, 128, 128])
fourier_model = dde.Model(data, fourier_net)

# --- Adam phase ---
fourier_model.compile("adam", lr=1e-3, loss_weights=[W_PDE, W_BC, W_IC])
losshistory, train_state = fourier_model.train(iterations=10000)

dde.utils.plot_loss_history(losshistory)
plt.title("Fourier PINN — Adam training loss")
plt.tight_layout()
plt.show()

# --- L-BFGS phase ---
fourier_model.compile("L-BFGS", loss_weights=[W_PDE, W_BC, W_IC])
losshistory, train_state = fourier_model.train()

dde.utils.plot_loss_history(losshistory)
plt.title("Fourier PINN — L-BFGS fine-tuning loss")
plt.tight_layout()
plt.show()

print("Fourier PINN training complete.")
```

- [ ] **Step 2: Run cell, verify**

Both loss plots appear. Loss decreases monotonically. No NaN values. Final total loss should be < 1e-4.

- [ ] **Step 3: Commit**

```bash
git add pinn_2d_fourier.ipynb
git commit -m "feat: train Fourier PINN (Adam + L-BFGS)"
```

---

### Task 10: Baseline FNN training on same IC

**Files:**
- Modify: `pinn_2d_fourier.ipynb` (add cell)

- [ ] **Step 1: Add baseline training cell**

```python
# Plain FNN baseline — same architecture as phase 1 (5 hidden layers × 32 neurons)
# Trained from scratch on the same multi-freq IC for fair comparison
baseline_net   = dde.nn.FNN([3] + [32] * 5 + [1], "tanh", "Glorot normal")
baseline_model = dde.Model(data, baseline_net)

# --- Adam phase ---
baseline_model.compile("adam", lr=1e-3, loss_weights=[W_PDE, W_BC, W_IC])
losshistory_b, _ = baseline_model.train(iterations=10000)

dde.utils.plot_loss_history(losshistory_b)
plt.title("Baseline FNN — Adam training loss")
plt.tight_layout()
plt.show()

# --- L-BFGS phase ---
baseline_model.compile("L-BFGS", loss_weights=[W_PDE, W_BC, W_IC])
losshistory_b, _ = baseline_model.train()

dde.utils.plot_loss_history(losshistory_b)
plt.title("Baseline FNN — L-BFGS fine-tuning loss")
plt.tight_layout()
plt.show()

print("Baseline FNN training complete.")
```

- [ ] **Step 2: Run cell, verify**

Both loss plots appear. Note whether the baseline loss is higher than the Fourier model at the same iteration count — this is the expected result.

- [ ] **Step 3: Commit**

```bash
git add pinn_2d_fourier.ipynb
git commit -m "feat: train baseline FNN on multi-freq IC"
```

---

### Task 11: Evaluation — heatmaps and comparison table

**Files:**
- Modify: `pinn_2d_fourier.ipynb` (add cell)

- [ ] **Step 1: Add evaluation cell**

```python
x_lin = np.linspace(0, 1, 100)
y_lin = np.linspace(0, 1, 100)
X, Y  = np.meshgrid(x_lin, y_lin)

print(f"{'t':>6} | {'Fourier L2':>12} | {'Baseline L2':>12} | {'Improvement':>12}")
print("-" * 52)

fig, axes = plt.subplots(len(T_EVAL), 3, figsize=(15, 4 * len(T_EVAL)))

for row, t_val in enumerate(T_EVAL):
    t_arr = np.full(X.shape, t_val)
    pts   = np.column_stack([X.ravel(), Y.ravel(), t_arr.ravel()])

    T_fourier  = fourier_model.predict(pts).reshape(100, 100)
    T_baseline = baseline_model.predict(pts).reshape(100, 100)
    T_true     = T_exact(X, Y, t_val)

    l2_fourier  = np.linalg.norm(T_fourier  - T_true) / np.linalg.norm(T_true)
    l2_baseline = np.linalg.norm(T_baseline - T_true) / np.linalg.norm(T_true)
    improvement = l2_baseline / l2_fourier  # >1 means Fourier is better

    print(f"{t_val:>6.3f} | {l2_fourier:>12.2e} | {l2_baseline:>12.2e} | {improvement:>11.1f}×")

    error = np.abs(T_fourier - T_true)
    vmin  = min(T_fourier.min(), T_true.min())
    vmax  = max(T_fourier.max(), T_true.max())

    im0 = axes[row, 0].imshow(T_fourier, origin="lower", extent=[0,1,0,1],
                               vmin=vmin, vmax=vmax, cmap="hot")
    axes[row, 0].set_title(f"Fourier PINN  t={t_val}")
    axes[row, 0].set_xlabel("x"); axes[row, 0].set_ylabel("y")
    plt.colorbar(im0, ax=axes[row, 0])

    im1 = axes[row, 1].imshow(T_true, origin="lower", extent=[0,1,0,1],
                               vmin=vmin, vmax=vmax, cmap="hot")
    axes[row, 1].set_title(f"Analytical  t={t_val}")
    axes[row, 1].set_xlabel("x"); axes[row, 1].set_ylabel("y")
    plt.colorbar(im1, ax=axes[row, 1])

    im2 = axes[row, 2].imshow(error, origin="lower", extent=[0,1,0,1],
                               cmap="viridis")
    axes[row, 2].set_title(f"|Error| (Fourier)  t={t_val}")
    axes[row, 2].set_xlabel("x"); axes[row, 2].set_ylabel("y")
    plt.colorbar(im2, ax=axes[row, 2])

plt.tight_layout()
plt.savefig("pinn_2d_fourier_results.png", dpi=150)
plt.show()
print("Evaluation complete. Figure saved to pinn_2d_fourier_results.png")
```

- [ ] **Step 2: Run cell, verify**

Table prints with three rows. Improvement column shows whether Fourier features help (>1×) or not. Heatmaps appear with three rows.

- [ ] **Step 3: Commit**

```bash
git add pinn_2d_fourier.ipynb pinn_2d_fourier_results.png
git commit -m "feat: evaluation — heatmaps and Fourier vs baseline comparison table"
```

---

## Self-Review

**Spec coverage:**
- [x] Multi-freq IC: `sin(πx)sin(πy) + 0.3·sin(5πx)sin(5πy)` — Task 3, 6, 7
- [x] Analytical solution with two decay modes — Task 3, 7
- [x] T_MAX=0.01, T_EVAL=[0.001, 0.005, 0.010] — Task 2
- [x] σ_bands=[1,5,20], m=64, 384 total features — Task 2, 8
- [x] MultiscaleFourierNet: frozen B matrices, sin+cos encoding, MLP 384→128→128→128→1 — Task 8
- [x] Adam 10k → L-BFGS for Fourier model — Task 9
- [x] Baseline FNN (5×32) trained on same IC — Task 10
- [x] Real test metric via solution_func — Task 7
- [x] Heatmaps + L2 error table + improvement ratio — Task 11

**Placeholder scan:** None found.

**Type consistency:** `fourier_model`, `baseline_model`, `T_exact`, `data`, `geomtime`, `pde`, `bc`, `ic`, `solution_func` all defined before use. `SIGMA_BANDS`, `M_FREQ`, `T_MAX`, `T_EVAL`, `W_PDE`, `W_BC`, `W_IC` defined in Task 2 and used consistently throughout.
