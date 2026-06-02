# PINN 2D Parametric Domain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single PINN in `pinn_2d_parametric.ipynb` that takes (x, y, t, Lx, Ly) as input and predicts the 2D heat equation solution for any rectangular plate with Lx, Ly ∈ [0.5, 2.0], without retraining.

**Architecture:** `ParametricFourierNet` (pure PyTorch, no DeepXDE) applies multi-scale Fourier features to normalized (x/Lx, y/Ly, t) coordinates and concatenates normalized (Lx, Ly) directly into the MLP. Training samples a fresh (Lx, Ly) pair every Adam step; L-BFGS fine-tunes on a fixed 3×3 geometry grid.

**Tech Stack:** Python 3.11, PyTorch 2.x, NumPy, Matplotlib. No DeepXDE.

---

## File Structure

```
pinn_2d_parametric.ipynb   # single new notebook, 11 cells
pinn_2d_parametric_results.png  # written by evaluation cell
```

---

### Task 1: Install, imports, and config cell

**Files:**
- Create: `pinn_2d_parametric.ipynb`

- [ ] **Step 1: Create notebook with cell 1 (install)**

Use `jupyter nbformat` or create the file. The easiest approach: open a terminal and run:

```bash
cd /Users/anjali/Projects/physics_deeplearning/vinci-assignment/pinn_heat
python3 -c "
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_code_cell('!pip install torch numpy matplotlib')]
with open('pinn_2d_parametric.ipynb', 'w') as f:
    nbf.write(nb, f)
print('created')
"
```

Expected: `created`

- [ ] **Step 2: Verify notebook exists**

```bash
ls -lh pinn_2d_parametric.ipynb
```

Expected: file exists, size > 0.

- [ ] **Step 3: Replace cell 1 with full install + imports + config**

Open `pinn_2d_parametric.ipynb` in the notebook editor and replace cell 1 with:

```python
!pip install torch numpy matplotlib
```

Then add a new code cell (cell 2) with:

```python
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
T_EVAL     = [0.001, 0.005, 0.010]
LX_RANGE   = [0.5, 2.0]
LY_RANGE   = [0.5, 2.0]

# Fourier feature config — same bands as phase 2
SIGMA_BANDS = [1, 3, 5]
M_FREQ      = 32  # random frequencies per band; total Fourier features = 3×2×32 = 192

N_ADAM  = 10000
N_LBFGS = 500

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

print(f"Device: {DEVICE}")
print(f"Fourier features: {len(SIGMA_BANDS)} bands × 2 × {M_FREQ} = {len(SIGMA_BANDS)*2*M_FREQ}")
print(f"MLP input size: {len(SIGMA_BANDS)*2*M_FREQ + 2}  (192 Fourier + 2 geometry)")
print("Config loaded.")
```

- [ ] **Step 4: Verify config cell runs without error**

In the notebook, run cell 2. Expected output:
```
Device: cpu
Fourier features: 3 bands × 2 × 32 = 192
MLP input size: 194  (192 Fourier + 2 geometry)
Config loaded.
```

- [ ] **Step 5: Commit**

```bash
git add pinn_2d_parametric.ipynb
git commit -m "feat: phase 3 notebook scaffold — install, imports, config"
```

---

### Task 2: Analytical solution and sanity checks

**Files:**
- Modify: `pinn_2d_parametric.ipynb` (add cell 3)

- [ ] **Step 1: Add cell 3 — analytical solution + checks**

```python
def T_exact(x, y, t, Lx, Ly, alpha=ALPHA):
    """
    Two-mode exact solution on [0,Lx]×[0,Ly]:
      mode 1: sin(πx/Lx)sin(πy/Ly) · exp(−λ₁t)
      mode 2: 0.3·sin(5πx/Lx)sin(5πy/Ly) · exp(−λ₂t)
    λ₁ = π²α(1/Lx² + 1/Ly²),  λ₂ = 25λ₁
    """
    lam1 = np.pi**2 * alpha * (1/Lx**2 + 1/Ly**2)
    lam2 = 25 * lam1
    low  = np.sin(np.pi * x / Lx) * np.sin(np.pi * y / Ly) * np.exp(-lam1 * t)
    high = 0.3 * np.sin(5*np.pi*x/Lx) * np.sin(5*np.pi*y/Ly) * np.exp(-lam2 * t)
    return low + high

# Sanity check 1: unit-square case matches phase 2 at t=0 (IC)
x0, y0 = np.array([0.5]), np.array([0.3])
val = T_exact(x0, y0, 0.0, 1.0, 1.0)
ic  = np.sin(np.pi*x0)*np.sin(np.pi*y0) + 0.3*np.sin(5*np.pi*x0)*np.sin(5*np.pi*y0)
assert np.isclose(val, ic).all(), f"Unit-square IC mismatch: {val} vs {ic}"
print(f"Sanity 1 passed: T_exact(0.5,0.3,0,1,1) = {val[0]:.6f}")

# Sanity check 2: BC is zero at x=0 for any Lx, Ly, t
val_bc = T_exact(np.array([0.0]), y0, 0.005, 1.5, 0.8)
assert np.isclose(val_bc, 0.0).all(), f"BC not zero: {val_bc}"
print(f"Sanity 2 passed: T_exact(0,y,t,Lx,Ly) = {val_bc[0]:.6f} (should be 0)")

# Sanity check 3: larger plate decays more slowly
lam1_small = np.pi**2 * ALPHA * (1/0.5**2 + 1/0.5**2)
lam1_large = np.pi**2 * ALPHA * (1/2.0**2 + 1/2.0**2)
assert lam1_small > lam1_large, "Small plate should decay faster"
print(f"Sanity 3 passed: λ₁(0.5×0.5)={lam1_small:.2f} > λ₁(2×2)={lam1_large:.2f}")

print("\nDecay rates at unit square (Lx=Ly=1):")
lam1 = np.pi**2 * ALPHA * 2
print(f"  λ₁ = {lam1:.4f}  (mode 1 half-life = {np.log(2)/lam1*1000:.2f} ms)")
print(f"  λ₂ = {25*lam1:.4f}  (mode 2 half-life = {np.log(2)/(25*lam1)*1000:.2f} ms)")
```

- [ ] **Step 2: Run cell 3, verify all three checks pass**

Expected:
```
Sanity 1 passed: T_exact(0.5,0.3,0,1,1) = <some float>
Sanity 2 passed: T_exact(0,y,t,Lx,Ly) = 0.000000 (should be 0)
Sanity 3 passed: λ₁(0.5×0.5)=157.91 > λ₁(2×2)=9.87
```

- [ ] **Step 3: Commit**

```bash
git add pinn_2d_parametric.ipynb
git commit -m "feat: analytical solution + sanity checks"
```

---

### Task 3: ParametricFourierNet class

**Files:**
- Modify: `pinn_2d_parametric.ipynb` (add cell 4)

- [ ] **Step 1: Add cell 4 — ParametricFourierNet**

```python
class ParametricFourierNet(nn.Module):
    """
    5D input: (x, y, t, Lx, Ly) in physical coordinates.
    Inside forward():
      - Normalize (x,y,t,Lx,Ly) to [-1,1]
      - Apply multi-scale Fourier features to (xi_norm, eta_norm, t_norm)
      - Concatenate (Lx_norm, Ly_norm) directly → 194-dim MLP input
    PyTorch autograd differentiates through the normalization automatically,
    so PDE residuals use standard ∂T/∂x, ∂T/∂y, ∂T/∂t.
    """
    def __init__(self, sigma_bands, m_freq, mlp_hidden, t_max):
        super().__init__()
        self.sigma_bands = sigma_bands
        self.t_max = t_max
        n_fourier = len(sigma_bands) * 2 * m_freq

        # Frozen random frequency matrices, one per band; shape (3, m_freq)
        for i, sigma in enumerate(sigma_bands):
            B = torch.randn(3, m_freq) * sigma
            self.register_parameter(f"B_{i}", nn.Parameter(B, requires_grad=False))

        # MLP: (n_fourier + 2) → hidden layers → 1
        sizes = [n_fourier + 2] + mlp_hidden + [1]
        layers = []
        for in_sz, out_sz in zip(sizes[:-2], sizes[1:-1]):
            layers += [nn.Linear(in_sz, out_sz), nn.Tanh()]
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x: (N, 5) — columns: [x_phys, y_phys, t, Lx, Ly]
        lx = x[:, 3:4]  # (N, 1)
        ly = x[:, 4:5]  # (N, 1)

        # Normalize physical coords to [-1, 1]
        xi_n  = 2.0 * x[:, 0:1] / lx - 1.0          # x/Lx rescaled
        eta_n = 2.0 * x[:, 1:2] / ly - 1.0           # y/Ly rescaled
        t_n   = 2.0 * x[:, 2:3] / self.t_max - 1.0   # t/T_MAX rescaled
        lx_n  = (lx - 1.25) / 0.75                    # Lx ∈ [0.5,2.0] → [-1,1]
        ly_n  = (ly - 1.25) / 0.75                    # Ly ∈ [0.5,2.0] → [-1,1]

        coords = torch.cat([xi_n, eta_n, t_n], dim=1)  # (N, 3) for Fourier

        # Multi-scale Fourier features: sin and cos projections per band
        features = []
        for i in range(len(self.sigma_bands)):
            B = getattr(self, f"B_{i}")   # (3, m_freq)
            proj = coords @ B              # (N, m_freq)
            features.append(torch.sin(proj))
            features.append(torch.cos(proj))

        # Concat Fourier features (192-dim) + geometry params (2-dim) = 194-dim
        z = torch.cat(features + [lx_n, ly_n], dim=-1)
        return self.mlp(z)
```

- [ ] **Step 2: Add shape test immediately after the class definition in the same cell**

Append to cell 4:

```python
# Shape test
_net = ParametricFourierNet(SIGMA_BANDS, M_FREQ, [128, 128, 128], T_MAX).to(DEVICE)
_x   = torch.rand(16, 5, device=DEVICE)
_x[:, 3] = 1.0   # Lx = 1.0
_x[:, 4] = 1.0   # Ly = 1.0
_x[:, 0] *= 1.0  # x ∈ [0, Lx]
_x[:, 1] *= 1.0  # y ∈ [0, Ly]
_x[:, 2] *= T_MAX
out = _net(_x)
assert out.shape == (16, 1), f"Expected (16,1), got {out.shape}"
print(f"Shape test passed: (16,5) → {out.shape}")

total_p    = sum(p.numel() for p in _net.parameters())
trainable  = sum(p.numel() for p in _net.parameters() if p.requires_grad)
print(f"Total parameters : {total_p:,}")
print(f"Trainable params : {trainable:,}  (frozen Fourier matrices excluded)")
del _net, _x, out
```

- [ ] **Step 3: Run cell 4, verify shape test passes**

Expected output (exact numbers will vary by hidden sizes):
```
Shape test passed: (16,5) → torch.Size([16, 1])
Total parameters : ~130,000
Trainable params : ~120,000  (frozen Fourier matrices excluded)
```

- [ ] **Step 4: Commit**

```bash
git add pinn_2d_parametric.ipynb
git commit -m "feat: ParametricFourierNet with shape test"
```

---

### Task 4: Collocation samplers and PDE residual

**Files:**
- Modify: `pinn_2d_parametric.ipynb` (add cell 5)

- [ ] **Step 1: Add cell 5 — samplers**

```python
def sample_interior(Lx, Ly, n, device):
    """N interior points in [0,Lx]×[0,Ly]×[0,T_MAX]. requires_grad=True for autograd."""
    x  = torch.rand(n, 1, device=device) * Lx
    y  = torch.rand(n, 1, device=device) * Ly
    t  = torch.rand(n, 1, device=device) * T_MAX
    lx = torch.full((n, 1), Lx, device=device)
    ly = torch.full((n, 1), Ly, device=device)
    pts = torch.cat([x, y, t, lx, ly], dim=1)
    pts.requires_grad_(True)
    return pts

def sample_boundary(Lx, Ly, n, device):
    """N boundary points on the four edges, t ~ U[0, T_MAX]. No requires_grad needed."""
    q = n // 4
    edges = []
    for x_val, y_samp, x_samp in [
        (0.0,   True,  False),   # x=0,  y random
        (Lx,    True,  False),   # x=Lx, y random
    ]:
        y  = torch.rand(q, 1, device=device) * Ly
        t  = torch.rand(q, 1, device=device) * T_MAX
        xc = torch.full((q, 1), x_val, device=device)
        edges.append(torch.cat([xc, y, t,
                                 torch.full((q,1), Lx, device=device),
                                 torch.full((q,1), Ly, device=device)], dim=1))
    for y_val in [0.0, Ly]:
        x  = torch.rand(q, 1, device=device) * Lx
        t  = torch.rand(q, 1, device=device) * T_MAX
        yc = torch.full((q, 1), y_val, device=device)
        edges.append(torch.cat([x, yc, t,
                                 torch.full((q,1), Lx, device=device),
                                 torch.full((q,1), Ly, device=device)], dim=1))
    return torch.cat(edges, dim=0)

def sample_ic(Lx, Ly, n, device):
    """N IC points: x∈[0,Lx], y∈[0,Ly], t=0."""
    x  = torch.rand(n, 1, device=device) * Lx
    y  = torch.rand(n, 1, device=device) * Ly
    t  = torch.zeros(n, 1, device=device)
    lx = torch.full((n, 1), Lx, device=device)
    ly = torch.full((n, 1), Ly, device=device)
    return torch.cat([x, y, t, lx, ly], dim=1)

def ic_target(pts):
    """IC values for points (N,5). Uses columns 0,1,3,4 for x,y,Lx,Ly."""
    x, y, lx, ly = pts[:,0:1], pts[:,1:2], pts[:,3:4], pts[:,4:5]
    low  = torch.sin(np.pi * x / lx) * torch.sin(np.pi * y / ly)
    high = 0.3 * torch.sin(5*np.pi*x/lx) * torch.sin(5*np.pi*y/ly)
    return low + high

def pde_residual(model, pts_pde):
    """
    PDE residual: ∂T/∂t − α(∂²T/∂x² + ∂²T/∂y²)
    pts_pde must have requires_grad=True (produced by sample_interior).
    """
    T = model(pts_pde)                                                     # (N,1)
    g1 = torch.autograd.grad(T.sum(), pts_pde, create_graph=True)[0]      # (N,5)
    T_x = g1[:, 0:1]
    T_y = g1[:, 1:2]
    T_t = g1[:, 2:3]
    T_xx = torch.autograd.grad(T_x.sum(), pts_pde, create_graph=True)[0][:, 0:1]
    T_yy = torch.autograd.grad(T_y.sum(), pts_pde, create_graph=True)[0][:, 1:2]
    return T_t - ALPHA * (T_xx + T_yy)
```

- [ ] **Step 2: Add quick sampler verification at the bottom of cell 5**

Append to cell 5:

```python
# Sampler shape tests
_Lx, _Ly = 1.5, 0.8
_pde = sample_interior(_Lx, _Ly, 10, DEVICE)
_bc  = sample_boundary(_Lx, _Ly, 20, DEVICE)
_ic  = sample_ic(_Lx, _Ly, 8, DEVICE)

assert _pde.shape == (10, 5) and _pde.requires_grad
assert _bc.shape  == (20, 5)
assert _ic.shape  == (8,  5)

# Boundary points should have x∈{0,Lx} or y∈{0,Ly}
x_bc = _bc[:, 0].cpu(); y_bc = _bc[:, 1].cpu()
on_boundary = ((x_bc == 0) | (x_bc == _Lx) | (y_bc == 0) | (y_bc == _Ly))
assert on_boundary.all(), "Some BC points are not on the boundary"

# IC points should have t=0
assert (_ic[:, 2] == 0).all(), "IC points must have t=0"

print("Sampler tests passed.")
print(f"  Interior: {_pde.shape}, requires_grad={_pde.requires_grad}")
print(f"  Boundary: {_bc.shape}")
print(f"  IC:       {_ic.shape}")
del _pde, _bc, _ic
```

- [ ] **Step 3: Run cell 5, verify all assertions pass**

Expected:
```
Sampler tests passed.
  Interior: torch.Size([10, 5]), requires_grad=True
  Boundary: torch.Size([20, 5])
  IC:       torch.Size([8, 5])
```

- [ ] **Step 4: Commit**

```bash
git add pinn_2d_parametric.ipynb
git commit -m "feat: collocation samplers and PDE residual function"
```

---

### Task 5: Adam training loop

**Files:**
- Modify: `pinn_2d_parametric.ipynb` (add cell 6)

- [ ] **Step 1: Instantiate model and add cell 6 — Adam loop**

Add a new cell (cell 6) with:

```python
model = ParametricFourierNet(SIGMA_BANDS, M_FREQ, [128, 128, 128], T_MAX).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
adam_losses = []

print(f"Training for {N_ADAM} Adam steps...")
for step in range(N_ADAM):
    # Sample a fresh geometry every step
    Lx = LX_RANGE[0] + torch.rand(1).item() * (LX_RANGE[1] - LX_RANGE[0])
    Ly = LY_RANGE[0] + torch.rand(1).item() * (LY_RANGE[1] - LY_RANGE[0])

    pts_pde = sample_interior(Lx, Ly, N_DOMAIN,   DEVICE)
    pts_bc  = sample_boundary(Lx, Ly, N_BOUNDARY, DEVICE)
    pts_ic  = sample_ic(Lx,      Ly, N_INITIAL,  DEVICE)

    optimizer.zero_grad()

    res      = pde_residual(model, pts_pde)
    loss_pde = (res ** 2).mean()

    T_bc     = model(pts_bc)
    loss_bc  = (T_bc ** 2).mean()

    T_ic_pred   = model(pts_ic)
    T_ic_true   = ic_target(pts_ic)
    loss_ic     = ((T_ic_pred - T_ic_true) ** 2).mean()

    loss = W_PDE * loss_pde + W_BC * loss_bc + W_IC * loss_ic
    loss.backward()
    optimizer.step()

    adam_losses.append(loss.item())
    if step % 1000 == 0:
        print(f"  step {step:5d} | loss={loss.item():.3e} "
              f"| pde={loss_pde.item():.3e} | bc={loss_bc.item():.3e} "
              f"| ic={loss_ic.item():.3e} | Lx={Lx:.2f} Ly={Ly:.2f}")

print(f"\nAdam complete. Final loss: {adam_losses[-1]:.3e}")
```

- [ ] **Step 2: Run cell 6 and verify loss decreases**

The run will take several minutes. Expected output pattern:
```
Training for 10000 Adam steps...
  step     0 | loss=X.XXe+XX | pde=... | bc=... | ic=...
  step  1000 | loss=...
  ...
  step  9000 | loss=...

Adam complete. Final loss: <lower than step 0>
```

The final loss should be lower than the step 0 loss. If the step-0 PDE loss is >1e4, something is wrong (check σ bands and normalization).

- [ ] **Step 3: Commit**

```bash
git add pinn_2d_parametric.ipynb
git commit -m "feat: Adam training loop with per-step geometry sampling"
```

---

### Task 6: L-BFGS fine-tuning and loss plots

**Files:**
- Modify: `pinn_2d_parametric.ipynb` (add cells 7 and 8)

- [ ] **Step 1: Add cell 7 — L-BFGS fine-tuning**

```python
# 3×3 grid of geometries for L-BFGS closure
LX_GRID = [0.5, 1.25, 2.0]
LY_GRID = [0.5, 1.25, 2.0]
LBFGS_GEOMS = [(lx, ly) for lx in LX_GRID for ly in LY_GRID]  # 9 pairs
N_EACH = max(1, N_DOMAIN // len(LBFGS_GEOMS))  # ~1111 interior pts per geometry

optimizer_lbfgs = torch.optim.LBFGS(
    model.parameters(), lr=1.0, max_iter=N_LBFGS,
    history_size=50, line_search_fn="strong_wolfe"
)
lbfgs_losses = []

def closure():
    optimizer_lbfgs.zero_grad()
    total = torch.tensor(0.0, device=DEVICE)
    for Lx, Ly in LBFGS_GEOMS:
        p_pde = sample_interior(Lx, Ly, N_EACH,              DEVICE)
        p_bc  = sample_boundary(Lx, Ly, N_BOUNDARY // len(LBFGS_GEOMS), DEVICE)
        p_ic  = sample_ic(Lx,      Ly, N_INITIAL  // len(LBFGS_GEOMS), DEVICE)

        res   = pde_residual(model, p_pde)
        l_pde = (res**2).mean()
        l_bc  = (model(p_bc)**2).mean()
        l_ic  = ((model(p_ic) - ic_target(p_ic))**2).mean()
        total = total + W_PDE * l_pde + W_BC * l_bc + W_IC * l_ic

    total = total / len(LBFGS_GEOMS)
    total.backward()
    lbfgs_losses.append(total.item())
    return total

print(f"Running L-BFGS (max_iter={N_LBFGS}) on {len(LBFGS_GEOMS)} fixed geometries...")
optimizer_lbfgs.step(closure)
print(f"L-BFGS complete. Loss: {lbfgs_losses[0]:.3e} → {lbfgs_losses[-1]:.3e} "
      f"({len(lbfgs_losses)} closure evaluations)")
```

- [ ] **Step 2: Add cell 8 — loss plots**

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.semilogy(adam_losses)
ax1.set_xlabel("Adam step")
ax1.set_ylabel("Total loss")
ax1.set_title("Adam training loss")
ax1.grid(True, which="both", alpha=0.3)

ax2.semilogy(lbfgs_losses)
ax2.set_xlabel("L-BFGS closure call")
ax2.set_ylabel("Total loss (avg over 9 geometries)")
ax2.set_title("L-BFGS fine-tuning loss")
ax2.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig("pinn_2d_parametric_loss.png", dpi=150)
plt.show()
print("Loss plots saved.")
```

- [ ] **Step 3: Run cells 7 and 8 and verify**

Expected: L-BFGS loss should be lower than the Adam final loss. The closure is called at least once. Loss plot shows downward trend in both phases.

- [ ] **Step 4: Commit**

```bash
git add pinn_2d_parametric.ipynb
git commit -m "feat: L-BFGS fine-tuning and loss plots"
```

---

### Task 7: Evaluation — heatmaps, L2 table, and regression check

**Files:**
- Modify: `pinn_2d_parametric.ipynb` (add cells 9 and 10)
- Create: `pinn_2d_parametric_results.png` (written by evaluation cell)

- [ ] **Step 1: Add cell 9 — evaluation loop**

```python
TEST_GEOMS = [
    ("Square (1×1)",    1.0, 1.0),
    ("Wide (1.5×0.7)",  1.5, 0.7),
    ("Tall (0.6×1.8)",  0.6, 1.8),
]

model.eval()

n_geoms = len(TEST_GEOMS)
n_times = len(T_EVAL)
fig, axes = plt.subplots(n_geoms * n_times, 3, figsize=(15, 4 * n_geoms * n_times))

print(f"{'Geometry':>18} | {'t':>6} | {'L2 error':>10}")
print("-" * 44)

row = 0
for name, Lx, Ly in TEST_GEOMS:
    for t_val in T_EVAL:
        x_lin = np.linspace(0, Lx, 100)
        y_lin = np.linspace(0, Ly, 100)
        X, Y  = np.meshgrid(x_lin, y_lin)

        pts_np = np.column_stack([
            X.ravel(), Y.ravel(),
            np.full(X.size, t_val),
            np.full(X.size, Lx),
            np.full(X.size, Ly),
        ])
        pts_t = torch.tensor(pts_np, dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            T_pred = model(pts_t).cpu().numpy().reshape(100, 100)

        T_true = T_exact(X, Y, t_val, Lx, Ly)
        l2 = np.linalg.norm(T_pred - T_true) / np.linalg.norm(T_true)
        print(f"{name:>18} | {t_val:>6.3f} | {l2:>10.2e}")

        error = np.abs(T_pred - T_true)
        vmin  = min(T_pred.min(), T_true.min())
        vmax  = max(T_pred.max(), T_true.max())

        im0 = axes[row, 0].imshow(T_pred, origin="lower",
                                   extent=[0, Lx, 0, Ly],
                                   vmin=vmin, vmax=vmax, cmap="hot")
        axes[row, 0].set_title(f"{name} PINN  t={t_val}")
        axes[row, 0].set_xlabel("x"); axes[row, 0].set_ylabel("y")
        plt.colorbar(im0, ax=axes[row, 0])

        im1 = axes[row, 1].imshow(T_true, origin="lower",
                                   extent=[0, Lx, 0, Ly],
                                   vmin=vmin, vmax=vmax, cmap="hot")
        axes[row, 1].set_title(f"{name} Analytical  t={t_val}")
        axes[row, 1].set_xlabel("x"); axes[row, 1].set_ylabel("y")
        plt.colorbar(im1, ax=axes[row, 1])

        im2 = axes[row, 2].imshow(error, origin="lower",
                                   extent=[0, Lx, 0, Ly], cmap="viridis")
        axes[row, 2].set_title(f"|Error|  t={t_val}")
        axes[row, 2].set_xlabel("x"); axes[row, 2].set_ylabel("y")
        plt.colorbar(im2, ax=axes[row, 2])

        row += 1

plt.tight_layout()
plt.savefig("pinn_2d_parametric_results.png", dpi=150)
plt.show()
print("\nEvaluation complete. Figure saved to pinn_2d_parametric_results.png")
```

- [ ] **Step 2: Add cell 10 — regression check against phase 2**

```python
# Phase 2 reference result (Lx=Ly=1, t=0.001): 0.60% relative L2 error
PHASE2_REFERENCE = 0.006  # 0.6%

x_lin = np.linspace(0, 1.0, 100)
y_lin = np.linspace(0, 1.0, 100)
X, Y  = np.meshgrid(x_lin, y_lin)
pts_np = np.column_stack([X.ravel(), Y.ravel(),
                           np.full(X.size, 0.001),
                           np.ones(X.size),
                           np.ones(X.size)])
pts_t = torch.tensor(pts_np, dtype=torch.float32, device=DEVICE)

with torch.no_grad():
    T_pred = model(pts_t).cpu().numpy().reshape(100, 100)
T_true = T_exact(X, Y, 0.001, 1.0, 1.0)
l2_unit = np.linalg.norm(T_pred - T_true) / np.linalg.norm(T_true)

print(f"Regression check — unit square (Lx=Ly=1) at t=0.001:")
print(f"  Phase 3 L2 error : {l2_unit:.2e}")
print(f"  Phase 2 reference: {PHASE2_REFERENCE:.2e}")
if l2_unit < 0.05:
    print("  PASS — parametric model solves the unit-square case (<5% L2)")
else:
    print("  WARN — unit-square error is high; consider more training")
```

- [ ] **Step 3: Run cells 9 and 10**

Expected from cell 9 (exact numbers depend on training, but should show the model works across geometries):
```
          Geometry |      t |   L2 error
--------------------------------------------
    Square (1×1) |  0.001 |   X.XXe-XX
    Square (1×1) |  0.005 |   X.XXe-XX
    Square (1×1) |  0.010 |   X.XXe-XX
  Wide (1.5×0.7) |  0.001 |   X.XXe-XX
  ...
```

Expected from cell 10:
```
Regression check — unit square (Lx=Ly=1) at t=0.001:
  Phase 3 L2 error : X.XXe-XX
  Phase 2 reference: 6.00e-03
  PASS — parametric model solves the unit-square case (<5% L2)
```

Note: Phase 3 may have higher error than phase 2 on the unit square — the model is harder (5 inputs, many geometries). A PASS threshold of 5% is intentionally generous.

- [ ] **Step 4: Commit**

```bash
git add pinn_2d_parametric.ipynb pinn_2d_parametric_results.png
git commit -m "feat: evaluation heatmaps, L2 table, and phase 2 regression check"
```

---

## Self-Review

**Spec coverage:**
- ✅ ParametricFourierNet (5D input, internal normalization, split encoding 192+2=194) — Task 3
- ✅ Continuous (Lx,Ly) sampling per Adam step — Task 5
- ✅ Custom PyTorch loop (no DeepXDE) — Tasks 4, 5, 6
- ✅ L-BFGS on 3×3 grid Lx,Ly ∈ {0.5, 1.25, 2.0} — Task 6
- ✅ Three held-out test geometries (1×1, 1.5×0.7, 0.6×1.8) — Task 7
- ✅ T_EVAL = [0.001, 0.005, 0.010] — Task 7
- ✅ Heatmaps (pred | analytical | error) — Task 7
- ✅ Relative L2 table — Task 7
- ✅ Regression check vs phase 2 — Task 7
- ✅ Analytical solution sanity checks — Task 2
- ✅ Config cell matches spec exactly — Task 1

**Placeholder scan:** None found.

**Type consistency:**
- `pde_residual(model, pts_pde)` — used consistently in Tasks 4, 5, 6
- `sample_interior`, `sample_boundary`, `sample_ic` — signature (Lx, Ly, n, device) consistent throughout
- `ic_target(pts)` — takes (N,5) tensor, consistent in Tasks 4, 5, 6
- `T_exact(x, y, t, Lx, Ly)` — numpy function, consistent in Tasks 2, 7
