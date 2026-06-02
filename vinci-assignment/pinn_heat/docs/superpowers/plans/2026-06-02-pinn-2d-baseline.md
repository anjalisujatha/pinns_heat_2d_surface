# PINN 2D Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a PINN in `pinn_2d.ipynb` that solves the transient 2D heat equation on a unit square and verifies against the known analytical solution.

**Architecture:** Plain MLP (5 hidden layers × 32 neurons, tanh) trained with DeepXDE's soft-constraint loss: PDE residual + Dirichlet BC + sinusoidal IC. Two-phase optimizer: Adam (10k iterations) → L-BFGS fine-tuning.

**Tech Stack:** Python, DeepXDE, PyTorch, NumPy, Matplotlib

---

## File Map

| File | Role |
|---|---|
| `pinn_2d.ipynb` | Single notebook containing all cells — install, config, model, training, evaluation |

All implementation lives in one notebook. Each task below corresponds to one notebook cell (or a tight group). Cells are appended in order.

---

### Task 1: Install dependencies

**Files:**
- Modify: `pinn_2d.ipynb` (add cell 1)

- [ ] **Step 1: Add and run install cell**

Add a code cell with:
```python
!pip install deepxde torch numpy matplotlib
```

- [ ] **Step 2: Verify**

Cell output ends with `Successfully installed deepxde-...` (or `already satisfied` if already installed). No red error lines.

- [ ] **Step 3: Commit**

```bash
git add pinn_2d.ipynb
git commit -m "feat: add install cell"
```

---

### Task 2: Imports and config

**Files:**
- Modify: `pinn_2d.ipynb` (add cell 2)

- [ ] **Step 1: Add imports and config cell**

```python
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import torch

dde.backend.set_default_backend("pytorch")

# --- Config ---
ALPHA   = 1.0    # thermal diffusivity
T_MAX   = 1.0    # simulation end time
N_DOMAIN   = 10000  # interior collocation points (LHS)
N_BOUNDARY = 800    # spatial boundary points (200 per edge)
N_INITIAL  = 500    # IC points at t=0
W_PDE = 1          # loss weight: PDE residual
W_BC  = 10         # loss weight: boundary condition
W_IC  = 10         # loss weight: initial condition
T_EVAL = [0.25, 0.50, 0.75]  # time snapshots for evaluation
```

- [ ] **Step 2: Run cell, verify**

Cell runs without error. DeepXDE prints a backend confirmation line like `Using backend: pytorch`.

- [ ] **Step 3: Commit**

```bash
git add pinn_2d.ipynb
git commit -m "feat: add imports and config"
```

---

### Task 3: Analytical solution

**Files:**
- Modify: `pinn_2d.ipynb` (add cell 3)

- [ ] **Step 1: Add analytical solution cell**

```python
def T_exact(x, y, t):
    """Exact solution: T = sin(πx)·sin(πy)·exp(−2π²αt)"""
    return np.sin(np.pi * x) * np.sin(np.pi * y) * np.exp(-2 * np.pi**2 * ALPHA * t)

# Quick sanity check: at t=0, T_exact should equal the IC
x_test = np.array([0.5])
y_test = np.array([0.5])
val_at_t0 = T_exact(x_test, y_test, 0.0)
val_ic    = np.sin(np.pi * 0.5) * np.sin(np.pi * 0.5)
print(f"T_exact(0.5, 0.5, 0) = {val_at_t0[0]:.6f}")
print(f"sin(π·0.5)² = {val_ic:.6f}")
assert np.isclose(val_at_t0[0], val_ic), "Analytical solution does not match IC"
print("Sanity check passed.")
```

- [ ] **Step 2: Run cell, verify**

Output:
```
T_exact(0.5, 0.5, 0) = 1.000000
sin(π·0.5)² = 1.000000
Sanity check passed.
```

- [ ] **Step 3: Commit**

```bash
git add pinn_2d.ipynb
git commit -m "feat: add analytical solution with sanity check"
```

---

### Task 4: Domain definition

**Files:**
- Modify: `pinn_2d.ipynb` (add cell 4)

- [ ] **Step 1: Add domain cell**

```python
# Spatial domain: unit square [0,1] x [0,1]
geom = dde.geometry.Rectangle(xmin=[0, 0], xmax=[1, 1])

# Time domain: [0, T_MAX]
timedomain = dde.geometry.TimeDomain(0, T_MAX)

# Combined space-time domain
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

print(f"Geometry: {geom}")
print(f"Time domain: [0, {T_MAX}]")
print("GeometryXTime created successfully.")
```

- [ ] **Step 2: Run cell, verify**

Cell runs without error and prints:
```
GeometryXTime created successfully.
```

- [ ] **Step 3: Commit**

```bash
git add pinn_2d.ipynb
git commit -m "feat: define space-time domain"
```

---

### Task 5: PDE residual function

**Files:**
- Modify: `pinn_2d.ipynb` (add cell 5)

- [ ] **Step 1: Add PDE residual cell**

Network input `x` has shape `(N, 3)` where columns are `[x_coord, y_coord, t]`.
Network output `u` has shape `(N, 1)`.

```python
def pde(x, u):
    """
    Residual of: ∂u/∂t - α(∂²u/∂x² + ∂²u/∂y²) = 0
    x[:, 0] = x_coord, x[:, 1] = y_coord, x[:, 2] = t
    """
    u_t  = dde.grad.jacobian(u, x, i=0, j=2)   # ∂u/∂t
    u_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)  # ∂²u/∂x²
    u_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)  # ∂²u/∂y²
    return u_t - ALPHA * (u_xx + u_yy)

print("PDE residual function defined.")
```

- [ ] **Step 2: Run cell, verify**

Prints: `PDE residual function defined.`

- [ ] **Step 3: Commit**

```bash
git add pinn_2d.ipynb
git commit -m "feat: define PDE residual using DeepXDE autograd"
```

---

### Task 6: Boundary and initial conditions

**Files:**
- Modify: `pinn_2d.ipynb` (add cell 6)

- [ ] **Step 1: Add BC and IC cell**

```python
# Dirichlet BC: T = 0 on all spatial boundaries for all t
def on_boundary(x, on_boundary):
    return on_boundary

bc = dde.icbc.DirichletBC(
    geomtime,
    func=lambda x: np.zeros((len(x), 1)),
    on_boundary=on_boundary,
)

# IC: T(x, y, 0) = sin(πx)·sin(πy)
def ic_func(x):
    return np.sin(np.pi * x[:, 0:1]) * np.sin(np.pi * x[:, 1:2])

ic = dde.icbc.IC(
    geomtime,
    func=ic_func,
    on_initial=lambda x, on_initial: on_initial,
)

print("Boundary condition (Dirichlet, T=0) defined.")
print("Initial condition (sin(πx)·sin(πy)) defined.")
```

- [ ] **Step 2: Run cell, verify**

Prints both confirmation lines without error.

- [ ] **Step 3: Commit**

```bash
git add pinn_2d.ipynb
git commit -m "feat: define Dirichlet BC and sinusoidal IC"
```

---

### Task 7: Dataset assembly

**Files:**
- Modify: `pinn_2d.ipynb` (add cell 7)

- [ ] **Step 1: Add TimePDE dataset cell**

DeepXDE uses LHS by default for `TimePDE`. The `loss_weights` order in compile will match `[pde, bc, ic]`.

```python
data = dde.data.TimePDE(
    geometryxtime=geomtime,
    pde=pde,
    ic_bcs=[bc, ic],
    num_domain=N_DOMAIN,
    num_boundary=N_BOUNDARY,
    num_initial=N_INITIAL,
    train_distribution="LHS",
)

print(f"Training dataset assembled:")
print(f"  Domain points : {N_DOMAIN}")
print(f"  Boundary pts  : {N_BOUNDARY}")
print(f"  IC points     : {N_INITIAL}")
```

- [ ] **Step 2: Run cell, verify**

Prints the dataset summary. DeepXDE may print its own sampling messages — that is expected.

- [ ] **Step 3: Commit**

```bash
git add pinn_2d.ipynb
git commit -m "feat: assemble TimePDE dataset with LHS sampling"
```

---

### Task 8: Network and model construction

**Files:**
- Modify: `pinn_2d.ipynb` (add cell 8)

- [ ] **Step 1: Add network cell**

```python
# FNN: [3, 32, 32, 32, 32, 32, 1] with tanh activations
layer_sizes = [3] + [32] * 5 + [1]
net = dde.nn.FNN(layer_sizes, "tanh", "Glorot normal")

model = dde.Model(data, net)

total_params = sum(p.numel() for p in net.parameters())
print(f"Network: {layer_sizes}")
print(f"Activation: tanh")
print(f"Total parameters: {total_params:,}")
```

- [ ] **Step 2: Run cell, verify**

Prints something like:
```
Network: [3, 32, 32, 32, 32, 32, 1]
Activation: tanh
Total parameters: 5,409
```

- [ ] **Step 3: Commit**

```bash
git add pinn_2d.ipynb
git commit -m "feat: define FNN and DeepXDE model"
```

---

### Task 9: Training — Adam phase

**Files:**
- Modify: `pinn_2d.ipynb` (add cell 9)

- [ ] **Step 1: Add Adam training cell**

Loss weights order matches `[pde, bc, ic]` — same order as `ic_bcs` list in `TimePDE`.

```python
model.compile(
    "adam",
    lr=1e-3,
    loss_weights=[W_PDE, W_BC, W_IC],
)

losshistory, train_state = model.train(iterations=10000)

# Plot loss curve
dde.utils.plot_loss_history(losshistory)
plt.title("Adam training loss")
plt.tight_layout()
plt.show()
print("Adam training complete.")
```

- [ ] **Step 2: Run cell, verify**

Training runs for 10,000 iterations. Loss history plot appears. Final total loss should be visibly decreasing. No NaN losses.

- [ ] **Step 3: Commit**

```bash
git add pinn_2d.ipynb
git commit -m "feat: Adam training phase (10k iterations)"
```

---

### Task 10: Training — L-BFGS fine-tuning

**Files:**
- Modify: `pinn_2d.ipynb` (add cell 10)

- [ ] **Step 1: Add L-BFGS cell**

```python
model.compile("L-BFGS", loss_weights=[W_PDE, W_BC, W_IC])
losshistory, train_state = model.train()

dde.utils.plot_loss_history(losshistory)
plt.title("L-BFGS fine-tuning loss")
plt.tight_layout()
plt.show()
print("L-BFGS fine-tuning complete.")
```

- [ ] **Step 2: Run cell, verify**

L-BFGS converges in well under 500 iterations. Final total loss is lower than after Adam. No NaN losses.

- [ ] **Step 3: Commit**

```bash
git add pinn_2d.ipynb
git commit -m "feat: L-BFGS fine-tuning phase"
```

---

### Task 11: Evaluation — plots and L2 error table

**Files:**
- Modify: `pinn_2d.ipynb` (add cell 11)

- [ ] **Step 1: Add evaluation cell**

```python
# Dense evaluation grid
x_lin = np.linspace(0, 1, 100)
y_lin = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x_lin, y_lin)

print(f"{'t':>6} | {'Rel. L2 error':>15}")
print("-" * 25)

fig, axes = plt.subplots(len(T_EVAL), 3, figsize=(15, 4 * len(T_EVAL)))

for row, t_val in enumerate(T_EVAL):
    # Build input array: (10000, 3)
    t_arr = np.full(X.shape, t_val)
    pts   = np.column_stack([X.ravel(), Y.ravel(), t_arr.ravel()])

    # Predict
    T_pred = model.predict(pts).reshape(100, 100)
    T_true = T_exact(X, Y, t_val)
    error  = np.abs(T_pred - T_true)

    # Relative L2 error
    l2 = np.linalg.norm(T_pred - T_true) / np.linalg.norm(T_true)
    print(f"{t_val:>6.2f} | {l2:>15.2e}")

    # Plots
    vmin = min(T_pred.min(), T_true.min())
    vmax = max(T_pred.max(), T_true.max())

    im0 = axes[row, 0].imshow(T_pred, origin="lower", extent=[0,1,0,1],
                               vmin=vmin, vmax=vmax, cmap="hot")
    axes[row, 0].set_title(f"PINN prediction  t={t_val}")
    axes[row, 0].set_xlabel("x"); axes[row, 0].set_ylabel("y")
    plt.colorbar(im0, ax=axes[row, 0])

    im1 = axes[row, 1].imshow(T_true, origin="lower", extent=[0,1,0,1],
                               vmin=vmin, vmax=vmax, cmap="hot")
    axes[row, 1].set_title(f"Analytical solution  t={t_val}")
    axes[row, 1].set_xlabel("x"); axes[row, 1].set_ylabel("y")
    plt.colorbar(im1, ax=axes[row, 1])

    im2 = axes[row, 2].imshow(error, origin="lower", extent=[0,1,0,1],
                               cmap="viridis")
    axes[row, 2].set_title(f"|Error|  t={t_val}")
    axes[row, 2].set_xlabel("x"); axes[row, 2].set_ylabel("y")
    plt.colorbar(im2, ax=axes[row, 2])

plt.tight_layout()
plt.savefig("pinn_2d_results.png", dpi=150)
plt.show()
print("Evaluation complete. Figure saved to pinn_2d_results.png")
```

- [ ] **Step 2: Run cell, verify**

Three rows of plots appear (one per t snapshot), each with: PINN prediction | Analytical | |Error|.

The error maps should be nearly uniform and small. Acceptable target: relative L2 error < 1% at all snapshots.

Console output should look like:
```
     t |   Rel. L2 error
-------------------------
  0.25 |        < 1e-2
  0.50 |        < 1e-2
  0.75 |        < 1e-2
```

- [ ] **Step 3: Commit**

```bash
git add pinn_2d.ipynb pinn_2d_results.png
git commit -m "feat: evaluation — heatmaps and L2 error table"
```

---

## Self-Review

**Spec coverage:**
- [x] DeepXDE on PyTorch
- [x] LHS sampling (10k interior, 800 boundary, 500 IC)
- [x] Sinusoidal IC, Dirichlet BC T=0
- [x] tanh activation, 5 × 32 MLP
- [x] Loss weights ω_pde=1, ω_bc=10, ω_ic=10
- [x] Adam 10k → L-BFGS
- [x] Visual heatmaps at t=0.25, 0.50, 0.75
- [x] Relative L2 error table

**Placeholder scan:** None found — all steps contain exact code.

**Type consistency:** `T_exact`, `pde`, `ic_func`, `bc`, `ic`, `data`, `net`, `model`, `geomtime` all defined before use. `ALPHA`, `T_MAX`, `W_PDE`, `W_BC`, `W_IC`, `T_EVAL` defined in Task 2 config cell and used consistently.