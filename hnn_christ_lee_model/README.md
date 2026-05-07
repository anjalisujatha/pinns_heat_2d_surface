# Hamiltonian Neural Network — Christ-Lee Model

A physics-informed neural network that learns the Hamiltonian of the Christ-Lee gauge theory model, with explicit enforcement of gauge invariance and symplectic structure via the Faddeev-Jackiw (FJ) formalism.

## Physics Background

The **Christ-Lee model** is a 2D gauge theory that serves as a canonical testbed for constrained Hamiltonian mechanics. It exhibits first-class constraints (gauge symmetries) and second-class constraints, making it an ideal system for studying the Faddeev-Jackiw reduction.

**Phase space**: `zeta = [r, P_r, theta, P_theta, lambda, rho]`
- `r`, `P_r`: radial coordinate and conjugate momentum
- `theta`, `P_theta`: angular coordinate and conjugate momentum
- `lambda`, `rho`: Lagrange multiplier and its conjugate (from BRST/FJ analysis)

**Primary constraint**: `P_theta = 0` (appears as a secondary constraint in polar coordinates)

**Gauge symmetry (zero mode)**:
```
delta(theta)  = +kappa
delta(lambda) = -kappa
```
The Hamiltonian `H` must be invariant under this transformation.

**Symplectic (FJ) brackets** used (from Eq. 28 of the paper):
```
{r, P_r} = 1
{theta, rho} = -1
{P_theta, lambda} = -1
{lambda, rho} = 1
```

## Model Architecture

`ChristLeeHNN` is a **Hamiltonian Neural Network (HNN)** — an MLP that learns the scalar Hamiltonian function `H(zeta)`, from which equations of motion are derived via automatic differentiation through the symplectic structure:

```
zeta_dot = J^{-1} @ (dH/d_zeta)
```

where `J` is the 6x6 inverse symplectic matrix encoding the FJ brackets.

**MLP architecture**: `6 → 256 → 256 → 1` with `Tanh` activations.

The symplectic matrix `J` is registered as a fixed non-trainable buffer, ensuring the FJ structure is exactly preserved during inference.

## Physics-Informed Loss

Four loss components are combined:

| Loss Term | Weight | Purpose |
|-----------|--------|---------|
| Dynamics MSE | 1.0 | Match numerically computed `zeta_dot` |
| Gauge invariance | 5.0 | `H(zeta)` must equal `H(gauge_shifted_zeta)` |
| Constraint | 0.5 | Penalize `P_theta != 0` (secondary constraint) |
| Consistency | 1.0 | Enforce `dot(lambda) + dot(theta) = 0` (zero-mode consistency) |

The gauge loss explicitly tests the symmetry `delta_theta = kappa`, `delta_lambda = -kappa` at training time.

## File Structure

```
hnn_christ_lee_model/
├── data/
│   ├── date_generator.py       # Generates analytic Christ-Lee trajectories (simple harmonic potential)
│   └── trajectory_generator.py # Alternative trajectory generator
├── models/
│   ├── hnn_core.py             # ChristLeeHNN: MLP + FJ symplectic structure
│   └── symplectic_utlis.py     # 6x6 inverse symplectic matrix J from FJ brackets
└── scripts/
    ├── train.py                # Training loop with gauge + constraint + dynamics loss
    └── verify_symmetry.py      # Tests gauge invariance: |H(zeta) - H(gauge_shifted_zeta)|
```

## Usage

**Train the model:**
```bash
cd hnn_christ_lee_model
python scripts/train.py
# Trains for 500 epochs, prints loss every 50 epochs
```

**Verify gauge symmetry:**
```bash
python scripts/verify_symmetry.py
# Prints: Gauge Symmetry Invariance Error (should approach 0 after training)
```

## Data Generation

Trajectories are generated analytically using a simple harmonic potential `V(r) = 0.5 * r^2`:
- `r(t) = 2.0 + 0.5 * cos(t)`
- `P_r(t) = -0.5 * sin(t)`
- Angular and multiplier variables initialized at zero (gauge-fixed slice)
- Numerical derivatives computed via `numpy.gradient` for supervision targets

## Key Physical Quantities

| Quantity | Index in state | Description |
|----------|---------------|-------------|
| `r` | 0 | Radial coordinate |
| `P_r` | 1 | Radial momentum |
| `theta` | 2 | Angular coordinate |
| `P_theta` | 3 | Angular momentum (constrained = 0) |
| `lambda` | 4 | Lagrange multiplier |
| `rho` | 5 | Multiplier's conjugate momentum |

## Verification

Run `verify_symmetry.py` to check that the trained Hamiltonian respects gauge invariance:
```
Gauge Symmetry Invariance Error: < 1e-4  (well-trained model)
```

## Dependencies

```
torch
numpy
```