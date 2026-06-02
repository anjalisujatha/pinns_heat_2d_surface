# PINN 2D Fourier Feature Embeddings — Design Spec
Date: 2026-06-02

## Goal
Test whether multi-scale Fourier feature embeddings outperform the plain MLP baseline (phase 1) on a harder, multi-frequency IC. The IC contains both a coarse mode (sin(πx)sin(πy)) and a fine mode (0.3·sin(5πx)sin(5πy)) — a problem where plain MLPs are known to suffer from spectral bias. This is phase 2 of the multi-scale PINN project.

## PDE & Domain
- **Equation:** ∂T/∂t = α(∂²T/∂x² + ∂²T/∂y²), α = 1.0
- **Domain:** (x, y) ∈ [0,1]², t ∈ [0, 0.01]
- **T_MAX = 0.01** — chosen so the high-frequency mode (decay constant 50π²) retains 18–61% of its IC amplitude across the evaluation window

**Boundary conditions (Dirichlet):** T = 0 on all four edges for all t > 0.

**Initial condition:**
```
T(x, y, 0) = sin(πx)·sin(πy) + 0.3·sin(5πx)·sin(5πy)
```

**Analytical solution:**
```
T_exact(x, y, t) = sin(πx)·sin(πy)·exp(−2π²t)
                 + 0.3·sin(5πx)·sin(5πy)·exp(−50π²t)
```

High-frequency mode amplitude at evaluation times:
- t=0.001: 0.3 × exp(−50π²×0.001) ≈ 0.3 × 0.61 = 0.18
- t=0.005: 0.3 × exp(−50π²×0.005) ≈ 0.3 × 0.085 = 0.026
- t=0.010: 0.3 × exp(−50π²×0.010) ≈ 0.3 × 0.0072 = 0.002

## Architecture: MultiscaleFourierNet

### Fourier Encoding
Maps raw input `v = (x, y, t)` → Fourier features before the MLP:

```
γ_σ(v) = [sin(B_σ · v), cos(B_σ · v)]
```
where `B_σ ~ N(0, σ²)`, shape `(3, m)`, sampled once at init and frozen.

Three frequency bands concatenated:

| Band | σ | Purpose |
|---|---|---|
| Low | 1 | Captures coarse sin(πx)sin(πy) mode |
| Mid | 5 | Captures fine sin(5πx)sin(5πy) mode |
| High | 20 | Captures sharp boundary gradients |

- **m = 64** frequencies per band
- Total Fourier features: 3 × 2 × 64 = **384**

### MLP on top of encoding
```
384 → 128 → 128 → 128 → 1
```
Activation: tanh throughout. Output: linear (scalar T).

### Implementation
`MultiscaleFourierNet` inherits from `dde.nn.NN` (PyTorch backend). The encoding is applied inside `forward()` before the MLP layers. Frozen `B` matrices registered as `torch.nn.Parameter(requires_grad=False)` so they move to the correct device automatically.

## Loss Function
```
L_total = ω_pde · MSE_pde + ω_bc · MSE_bc + ω_ic · MSE_ic
```

| Term | Points | Enforces |
|---|---|---|
| MSE_pde | 10,000 LHS in [0,1]²×[0,0.01] | ∂T/∂t − α∇²T = 0 |
| MSE_bc | 800 on spatial boundaries | T = 0 |
| MSE_ic | 500 at t=0 | T = sin(πx)sin(πy) + 0.3·sin(5πx)sin(5πy) |

**Weights:** ω_pde=1, ω_bc=10, ω_ic=10

## Training
| Stage | Optimizer | Config |
|---|---|---|
| 1 | Adam | lr=1e-3, 10,000 iterations |
| 2 | L-BFGS | default, until convergence |

## Evaluation
- **Snapshots:** t = 0.001, 0.005, 0.010
- **Visual:** Heatmaps — PINN prediction | Analytical | |Error| at each snapshot
- **Quantitative:** Relative L2 error `‖T_pred − T_exact‖ / ‖T_exact‖` on 100×100 grid
- **Comparison:** Side-by-side L2 error table — Fourier net vs plain FNN baseline. The baseline FNN (same 5×32 architecture as phase 1) is trained from scratch on the multi-freq IC within this notebook so the comparison is fair (same IC, same collocation points, same optimizer schedule).

## Notebook Layout (`pinn_2d_fourier.ipynb`)
1. Install & imports
2. Config (σ_bands=[1,5,20], m=64, T_MAX=0.01, T_EVAL=[0.001,0.005,0.010])
3. Multi-freq IC + analytical solution (with per-mode sanity check)
4. Domain definition (GeometryXTime, identical to baseline)
5. PDE residual (identical to baseline)
6. BC and IC definitions (IC target updated)
7. Dataset assembly (TimePDE + solution_func for real test metric)
8. `MultiscaleFourierNet` class definition
9. Model construction + parameter count
10. Training (Adam → L-BFGS) + loss plots
11. Evaluation: heatmaps + L2 error table + baseline comparison

## Out of Scope (this phase)
- Learnable Fourier frequencies
- Residual-Based Adaptive Refinement (RAR)
- 3D geometry / NVIDIA Modulus (phase 3)
- Parameterized domain dimensions (phase 3)
