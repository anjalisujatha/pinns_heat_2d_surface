# PINN 2D Baseline — Design Spec
Date: 2026-06-02

## Goal
Train a Physics-Informed Neural Network (PINN) to solve the transient 2D heat equation on a unit square with sinusoidal initial conditions. Verify correctness both visually and quantitatively against the known analytical solution. This is the first phase of a multi-scale PINN project; the architecture is intentionally simple so that multi-scale extensions (Fourier features, multi-branch networks) can be added incrementally.

## PDE & Domain
- **Equation:** ∂T/∂t = α(∂²T/∂x² + ∂²T/∂y²)
- **Domain:** (x, y) ∈ [0,1]², t ∈ [0, T_max]
- **Diffusivity:** α = 1.0 (normalized)
- **T_max:** 0.2

**Boundary conditions (Dirichlet):** T = 0 on all four edges for all t.

**Initial condition:** T(x, y, 0) = sin(πx)·sin(πy)

**Analytical solution:** T_exact(x, y, t) = sin(πx)·sin(πy)·exp(−2π²αt)

## Architecture
- **Input:** (x, y, t) — 3 scalars
- **Hidden layers:** 5 layers, 64 neurons each
- **Activation:** tanh (smooth, required for clean second-order autograd)
- **Output:** 1 scalar (T), linear activation

## Loss Functions
```
L_total = λ_pde · L_physics + λ_bc · L_bc + λ_ic · L_ic
```

| Term | Sampled from | Enforces |
|---|---|---|
| `L_physics` | random (x,y,t) ∈ [0,1]²×[0,T_max] | ∂T/∂t − α∇²T = 0 |
| `L_bc` | random points on all 4 edges × [0,T_max] | T = 0 on boundary |
| `L_ic` | random (x,y) ∈ [0,1]² at t=0 | T = sin(πx)sin(πy) |

**Weights:** λ_pde=1, λ_bc=10, λ_ic=10

PDE gradients computed via `torch.autograd.grad` with `create_graph=True`.

## Collocation Sampling
| Set | Count | Notes |
|---|---|---|
| Interior (PDE) | 2000 | Random uniform in [0,1]²×[0,T_max] |
| Boundary (BC) | 200 per edge (800 total) | Random uniform on each of 4 edges |
| Initial condition | 500 | Random uniform in [0,1]² at t=0 |

Points resampled each epoch to avoid overfitting to a fixed grid.

## Training
| Stage | Optimizer | Steps | Purpose |
|---|---|---|---|
| 1 | Adam, lr=1e-3 | 5000 | Fast initial convergence |
| 2 | L-BFGS | 500 | Fine-tune to low residual |

## Evaluation
- **Visual:** Side-by-side heatmaps of T_pred vs T_exact at t = 0.0, 0.05, 0.1, 0.2
- **Quantitative:** Relative L2 error `‖T_pred − T_exact‖ / ‖T_exact‖` on a 100×100 grid at each snapshot

## Notebook Layout (`pinn_2d.ipynb`)
1. Imports & config (α, domain, collocation counts, loss weights)
2. Analytical solution + collocation samplers
3. MLP model definition
4. Loss functions (each as a named function)
5. Training loop (Adam → L-BFGS)
6. Plots + L2 error table

## Out of Scope (this phase)
- Fourier feature embeddings (phase 2)
- Parameterized domain dimensions (phase 3)
- 3D geometry (phase 4)
- Adaptive collocation point sampling
- Parameterized BCs or ICs