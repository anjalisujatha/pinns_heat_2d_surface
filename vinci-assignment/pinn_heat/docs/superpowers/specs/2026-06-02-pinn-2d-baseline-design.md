# PINN 2D Baseline — Design Spec
Date: 2026-06-02

## Goal
Train a Physics-Informed Neural Network (PINN) to solve the transient 2D heat equation on a unit square with sinusoidal initial conditions. Verify correctness both visually and quantitatively against the known analytical solution. This is phase 1 of a multi-scale PINN project; the architecture is intentionally simple so multi-scale extensions can be added incrementally. Phase 3 (3D geometry) will migrate to NVIDIA Modulus.

## PDE & Domain
- **Equation:** ∂T/∂t = α(∂²T/∂x² + ∂²T/∂y²)
- **Domain:** (x, y) ∈ [0,1]², t ∈ [0, 1.0]
- **Diffusivity:** α = 1.0 (normalized)

**Boundary conditions (Dirichlet):** T = 0 on all four edges for all t > 0.

**Initial condition:** T(x, y, 0) = sin(πx)·sin(πy)

**Analytical solution:** T_exact(x, y, t) = sin(πx)·sin(πy)·exp(−2π²αt)

## Libraries & Tools
- **Deep learning backend:** PyTorch
- **PINN framework:** DeepXDE (built on PyTorch) — handles geometry, LHS sampling, AD-based PDE residuals, and two-phase training out of the box
- **Note for phase 3:** Migrate to NVIDIA Modulus (CSG support for complex 3D engineering geometry)

## Architecture
- **Input:** (x, y, t) — 3 scalars via DeepXDE's `GeometryXTime`
- **Hidden layers:** 4–9 layers, 20–50 neurons each (start with 5 × 32 for fast iteration)
- **Activation:** `tanh` — infinitely differentiable, prevents numerical stiffness in second-order AD
- **Output:** 1 scalar (T), linear activation

## Loss Function
```
L_total = ω_pde · MSE_pde + ω_bc · MSE_bc + ω_ic · MSE_ic
```

| Term | Points sampled from | Enforces |
|---|---|---|
| `MSE_pde` | 10,000 LHS points in [0,1]²×[0,1] | ∂T/∂t − α(∂²T/∂x² + ∂²T/∂y²) = 0 |
| `MSE_bc` | ~200 points per edge × [0,1] | T = 0 on all 4 boundaries |
| `MSE_ic` | ~500 points in [0,1]² at t=0 | T = sin(πx)sin(πy) |

**Weights:** ω_pde=1, ω_bc=10, ω_ic=10

PDE derivatives computed via DeepXDE's built-in Jacobian/Hessian (wraps torch.autograd).

## Collocation Sampling
- **Strategy:** Latin Hypercube Sampling (LHS) for even distribution across x, y, t
- **Interior:** 10,000 points
- **Boundary:** ~200 per edge (800 total)
- **IC:** 500 points at t=0

## Training
| Stage | Optimizer | Config | Purpose |
|---|---|---|---|
| 1 | Adam | lr=1e-3, ~10,000 iterations | Fast escape from bad local minima |
| 2 | L-BFGS | default config, ~500 iterations | Fine-tune to high-accuracy local minimum |

DeepXDE's `Model.train()` handles the two-phase handoff natively.

## Evaluation
- **Visual:** Side-by-side heatmaps of T_pred vs T_exact at t = 0.25, 0.50, 0.75
- **Quantitative:** Relative L2 error `‖T_pred − T_exact‖ / ‖T_exact‖` on a 100×100 grid at each snapshot

## Notebook Layout (`pinn_2d.ipynb`)
1. Install & imports (DeepXDE, PyTorch, numpy, matplotlib)
2. Config block (α, T_max, collocation counts, loss weights)
3. Analytical solution function
4. Domain definition (`dde.geometry.Rectangle` + `dde.geometry.TimeDomain` → `GeometryXTime`)
5. PDE residual function (using `dde.grad.jacobian` / `dde.grad.hessian`)
6. BC and IC definitions
7. `dde.data.TimePDE` dataset assembly
8. Network and model construction
9. Training (Adam → L-BFGS)
10. Plots + L2 error table

## Out of Scope (this phase)
- Fourier feature embeddings (phase 2)
- Parameterized domain dimensions (phase 3)
- 3D geometry / NVIDIA Modulus (phase 4)
- Residual-Based Adaptive Refinement (RAR) — available in DeepXDE, deferred to phase 2