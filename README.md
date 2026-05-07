# Physics-Based Deep Learning

A collection of deep learning models grounded in classical mechanics, differential geometry, and mathematical physics. Each project encodes physical structure directly into the model architecture or loss function — symmetries, conservation laws, and topological constraints are not approximated but enforced.

## Projects

| Project | Description |
|---|---|
| [`g_gnn_torus_knot/`](g_gnn_torus_knot/README.md) | Equivariant GNN that learns the constrained dynamics of a particle on a torus knot using the Faddeev-Jackiw symplectic formalism |
| [`hnn_christ_lee_model/`](hnn_christ_lee_model/README.md) | Hamiltonian Neural Network for the Christ-Lee gauge theory model with explicit gauge-invariance enforcement |
| [`thermal_nn/`](thermal_nn/README.md) | 3D Tensorized Fourier Neural Operator (TFNO) for thermal field prediction, trained with a physics-informed residual loss derived from Fourier's law |

## Common Themes

- **Constrained dynamics**: all three models operate on systems with non-trivial constraint surfaces or gauge redundancies
- **Symplectic structure**: Hamiltonian/Faddeev-Jackiw frameworks are used to derive equations of motion that the models must respect
- **Physics-informed losses**: beyond MSE, each model includes loss terms that penalize violation of conservation laws, topological constraints, or PDE residuals
- **Differentiable physics**: gradients flow through physical quantities (energy, symplectic structure, PDE operators) during training

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torch_geometric e3nn neuralop scipy matplotlib
```

Each sub-project is self-contained and can be run independently. See the individual READMEs for details.