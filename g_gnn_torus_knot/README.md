# Equivariant GNN for Torus Knot Dynamics

An E(3)-equivariant Graph Neural Network that learns the constrained dynamics of a particle moving along a torus knot, derived from the Faddeev-Jackiw symplectic formulation.

## Physics Background

A torus knot is a closed curve that winds around a torus `p` times in the toroidal direction and `q` times in the poloidal direction (where `gcd(p, q) = 1`). This project studies a free particle constrained to the `(2,3)`-torus knot embedded in toroidal coordinates `(eta, theta, phi)` using the Bipolar coordinate system.

The equations of motion are derived using the **Faddeev-Jackiw (FJ) method** — a first-order symplectic formalism that handles constraints without introducing Lagrange multipliers. The FJ brackets replace Poisson brackets and directly encode the constrained phase space geometry.

**State vector**: `[theta, phi, P_theta, P_phi]`

**Topological constraint** (knot condition):
```
p * theta + q * phi = 0  (mod 2*pi)
```

## Model Architecture

The model uses [`e3nn`](https://e3nn.org/)'s `gate_points_2101` equivariant network, which respects the full E(3) symmetry group (rotations, reflections, and translations in 3D space). This is physically motivated: the knot dynamics should transform covariantly under spatial rotations.

**Irreducible representations used:**
- Input: `1x0e + 1x1o` — scalar (mass) + vector (3D position)
- Hidden: `32x0e + 16x1o + 8x2e` — scalars, vectors, and symmetric 2-tensors
- Output: `1x1o` — predicted next position (vector)

**Edge attributes**: spherical harmonics of inter-node displacement vectors, enabling the network to reason about local geometry on the knot.

## Physics-Informed Loss

Three loss components are combined during training:

| Loss Term | Weight | Purpose |
|-----------|--------|---------|
| MSE reconstruction | 1.0 | Match ground truth next positions |
| Topological constraint | 100.0 | Force predictions to stay on the `(p,q)` knot |
| Hamiltonian regularization | 10.0 | Penalize kinetic energy drift (energy conservation) |

The high weight on the topological constraint (100.0) directly penalizes the knot residual `|p*theta + q*phi|`.

## File Structure

```
g_gnn_torus_knot/
├── data/
│   ├── generator.py            # Integrates FJ equations of motion via RK45, outputs toroidal + Cartesian trajectories
│   └── torus_knot_data.pt      # Generated trajectory data (2000 timesteps)
├── model/
│   └── architecture.py         # Defines the e3nn equivariant GNN
├── scripts/
│   ├── train.py                # Training loop with physics-informed loss
│   └── verify_model.py         # Autoregressive rollout, knot residual metric, equivariance audit, 3D plot
└── trained_torus_gnn.pth       # Saved model weights
```

## Usage

**Step 1: Generate training data**
```bash
cd g_gnn_torus_knot
python data/generator.py
# Outputs: data/torus_knot_data.pt
```

**Step 2: Train the model**
```bash
python scripts/train.py
# Trains for 1000 epochs, outputs: trained_torus_gnn.pth
```

**Step 3: Verify and visualize**
```bash
python scripts/verify_model.py
# Prints: Mean Knot Residual, Equivariance Error
# Shows: 3D plot of ground truth vs. GNN rollout
```

## Physical Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `R` | 3.0 | Major radius of torus |
| `r` | 1.0 | Minor radius of torus |
| `p, q` | 2, 3 | Knot winding numbers |
| `eta` | `arccosh(R/r)` | Fixed toroidal surface |
| `m` | 1.0 | Particle mass |

## Verification Metrics

- **Knot Residual**: `mean(|p*theta + q*phi|)` over a 500-step autoregressive rollout — measures how far predictions drift off the knot
- **Equivariance Error**: `mean(|f(Rx) - R*f(x)|)` for a 90° rotation `R` — verifies the E(3) symmetry guarantee

## Dependencies

```
torch
torch_geometric
e3nn
scipy
matplotlib
numpy
```