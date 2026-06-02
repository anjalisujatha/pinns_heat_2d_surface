# PINN 2D Parametric Domain — Design Spec
Date: 2026-06-02

## Goal
Train a single PINN that generalizes across rectangular plate geometries of varying size. The model takes (x, y, t, Lx, Ly) as input and predicts the temperature field without retraining, for any plate with Lx, Ly ∈ [0.5, 2.0]. This is phase 3 of the multi-scale PINN project.

## PDE & Domain
- **Equation:** ∂T/∂t = α(∂²T/∂x² + ∂²T/∂y²), α = 1.0
- **Physical domain:** x ∈ [0, Lx], y ∈ [0, Ly], t ∈ [0, T_MAX=0.01]
- **Geometry parameters:** Lx, Ly ~ Uniform[0.5, 2.0], sampled continuously during training

**Boundary conditions (Dirichlet):** T = 0 on all four edges for all t > 0.

**Initial condition:**
```
T(x, y, 0) = sin(πx/Lx)·sin(πy/Ly) + 0.3·sin(5πx/Lx)·sin(5πy/Ly)
```

**Analytical solution:**
```
T(x, y, t) = sin(πx/Lx)·sin(πy/Ly)·exp(−λ₁t) + 0.3·sin(5πx/Lx)·sin(5πy/Ly)·exp(−λ₂t)
λ₁ = π²α(1/Lx² + 1/Ly²)
λ₂ = 25π²α(1/Lx² + 1/Ly²)
```

The decay rates depend on Lx and Ly — a larger plate decays more slowly. The unit-square case (Lx=Ly=1) reduces exactly to the phase 2 problem.

**Note on coordinates:** The PDE is kept in standard physical form (∂T/∂t = α(∂²T/∂x² + ∂²T/∂y²)) throughout. Normalization to [−1,1] is an internal operation inside the network's `forward()` — PyTorch autograd propagates the chain rule automatically, so no coordinate Jacobians appear in the training code.

## Architecture: ParametricFourierNet

### Input preprocessing (5D → normalized 5D)
| Raw input | Range | Normalization | Normalized range |
|---|---|---|---|
| x | [0, Lx] | ξ_norm = 2x/Lx − 1 | [−1, 1] |
| y | [0, Ly] | η_norm = 2y/Ly − 1 | [−1, 1] |
| t | [0, T_MAX] | t_norm = 2t/T_MAX − 1 | [−1, 1] |
| Lx | [0.5, 2.0] | Lx_norm = (Lx − 1.25)/0.75 | [−1, 1] |
| Ly | [0.5, 2.0] | Ly_norm = (Ly − 1.25)/0.75 | [−1, 1] |

### Split encoding
```
(ξ_norm, η_norm, t_norm)  →  Fourier features  →  192-dim
                                                         │
(Lx_norm, Ly_norm)  ─────────────────────────────────── ┤
                                                         ↓
                                                    194-dim concat
                                                         │
                                                    MLP 194 → 128 → 128 → 128 → 1
```

Fourier features: same three bands σ=[1,3,5], m=32 random frequencies per band (frozen). Total: 3 × 2 × 32 = 192 features. Geometry parameters (Lx_norm, Ly_norm) enter the MLP linearly at full precision — not encoded via Fourier features, since domain size is not an oscillatory quantity.

MLP: tanh activations throughout, linear output.

`ParametricFourierNet` inherits from `torch.nn.Module` (no DeepXDE dependency — see Training section). Frozen B matrices registered as `nn.Parameter(requires_grad=False)`.

## Loss Function
```
L_total = W_PDE · MSE_pde + W_BC · MSE_bc + W_IC · MSE_ic
```

| Term | Points per step | Enforces |
|---|---|---|
| MSE_pde | 10,000 in [0,Lx]×[0,Ly]×[0,T_MAX] | ∂T/∂t − α(∂²T/∂x² + ∂²T/∂y²) = 0 |
| MSE_bc | 800 on four edges (x∈{0,Lx} or y∈{0,Ly}) | T = 0 |
| MSE_ic | 500 at t=0, x∈[0,Lx], y∈[0,Ly] | T = sin(πx/Lx)sin(πy/Ly) + 0.3·sin(5πx/Lx)sin(5πy/Ly) |

**Weights:** W_PDE=1, W_BC=10, W_IC=10

## Training

### Why a custom PyTorch loop (not DeepXDE)
DeepXDE geometry objects have fixed bounds. Continuously sampling (Lx, Ly) per iteration requires generating fresh collocation points each step — this is straightforward with raw PyTorch autograd but awkward to force into DeepXDE's abstraction. The custom loop is ~60 lines.

### Per-step procedure
1. Sample (Lx, Ly) ~ Uniform[0.5, 2.0]²
2. Sample N_DOMAIN=10,000 interior points: x ~ Uniform[0,Lx], y ~ Uniform[0,Ly], t ~ Uniform[0, T_MAX]
3. Sample N_BOUNDARY=800 boundary points: random edges (x∈{0,Lx} or y∈{0,Ly}), random t
4. Sample N_INITIAL=500 IC points: x ~ Uniform[0,Lx], y ~ Uniform[0,Ly], t=0
5. Forward pass network on each set; compute PDE residual via `torch.autograd.grad`
6. Compute L_total, backprop, Adam step

### Optimizer schedule
| Stage | Optimizer | Config |
|---|---|---|
| 1 | Adam | lr=1e-3, 10,000 iterations |
| 2 | L-BFGS | `torch.optim.LBFGS`, max_iter=500, fixed (Lx, Ly) grid of 9 sizes (3×3 over [0.5,2.0]) |

L-BFGS uses a fixed set of geometries since it requires a closure (repeated forward passes). The 3×3 grid covers Lx, Ly ∈ {0.5, 1.25, 2.0}.

## Evaluation

Three held-out test geometries (not in the L-BFGS grid):

| Geometry | Lx | Ly |
|---|---|---|
| Square (sanity check) | 1.0 | 1.0 |
| Wide plate | 1.5 | 0.7 |
| Tall plate | 0.6 | 1.8 |

For each geometry at T_EVAL = [0.001, 0.005, 0.010]:
- Heatmaps: PINN prediction | Analytical | |Error| (3-column layout, same as phase 2)
- Relative L2 error `‖T_pred − T_exact‖ / ‖T_exact‖` on 100×100 grid in (ξ, η)

Summary table: 3 geometries × 3 times = 9 relative L2 errors.

The Lx=Ly=1 row serves as a regression check against phase 2 (target: <1% at t=0.001).

## Notebook Layout (`pinn_2d_parametric.ipynb`)
1. Install & imports
2. Config (SIGMA_BANDS=[1,3,5], M_FREQ=32, T_MAX=0.01, LX_RANGE=[0.5,2.0], LY_RANGE=[0.5,2.0], T_EVAL=[0.001,0.005,0.010])
3. Analytical solution + sanity checks (unit-square case matches phase 2)
4. `ParametricFourierNet` class definition
5. Collocation samplers (interior, boundary, IC)
6. PDE residual via autograd
7. Training loop (Adam phase)
8. L-BFGS fine-tuning
9. Loss plots
10. Evaluation: heatmaps + L2 table for all three test geometries
11. Regression check: unit-square L2 vs phase 2 result

## Out of Scope (this phase)
- 3D geometry (phase 4)
- NVIDIA Modulus
- Learnable Fourier frequencies
- Parameterized α (thermal diffusivity)
- Non-Dirichlet boundary conditions
