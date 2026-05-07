# Thermal Neural Operator (3D TFNO + PINO)

A 3D Tensorized Fourier Neural Operator (TFNO) that learns to predict temperature fields from spatially varying thermal conductivity and heat source inputs, trained with a Physics-Informed Neural Operator (PINO) loss derived from Fourier's law.

## Physics Background

The governing equation is the **steady-state heat equation** (Fourier's law of heat conduction):

```
-div(kappa * grad(T)) = Q
```

where:
- `kappa(x)` — spatially varying thermal conductivity field (W/m·K)
- `Q(x)` — volumetric heat source (W/m³)
- `T(x)` — temperature field (K) to be predicted

The model maps `(kappa, Q) -> T` on a 3D spatial grid, learning the solution operator of the PDE rather than a fixed-resolution solution.

**Motivating application**: rapid thermal simulation of electronic chip packages — materials like silicon (`kappa ≈ 150 W/m·K`) are embedded in air (`kappa ≈ 0.026 W/m·K`) with localized heat sources.

## Model Architecture

The model uses a **Tensorized Fourier Neural Operator (TFNO)** from the [`neuralop`](https://neuraloperator.github.io/) library, with Tucker decomposition applied to the spectral weight tensors for memory efficiency.

**Key hyperparameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_modes` | (12, 12, 12) | Fourier frequency modes retained per spatial dimension |
| `hidden_channels` | 32 | Latent channel width |
| `in_channels` | 2 | Input: `[kappa, Q]` stacked as channels |
| `out_channels` | 1 | Output: temperature field `T` |
| `factorization` | tucker | Tucker decomposition for spectral weight compression |
| `rank` | 0.4 | Compression rank (40% of full-rank parameters) |

The TFNO applies learned Fourier integral operators in spectral space, enabling zero-shot generalization to higher spatial resolutions at inference time ("zero-shot scaling to 128³").

## Physics-Informed Loss (PINO)

The training loss combines a data term and a PDE residual term:

```
L_total = L_data + 0.1 * L_physics
```

**Data loss**: standard MSE between predicted and ground truth temperature fields.

**Physics residual** (`pino_loss`): computed entirely in PyTorch via `torch.gradient`, without any external PDE solver at training time:

1. Compute `grad(T_pred)` using finite differences
2. Compute flux: `F = -kappa * grad(T_pred)`
3. Compute divergence: `div(F)`
4. Residual: `div(F) + Q` (should be zero by the PDE)
5. Loss: `mean((div(F) + Q)^2)`

This directly penalizes violations of Fourier's law during training.

## File Structure

```
thermal_nn/
├── main.py                 # Training loop: loads data, builds model, runs PINO training
├── model/
│   ├── tfno_3d.py          # TFNO model definition (Tucker-factorized)
│   └── layers.py           # (Reserved for custom layer definitions)
├── data/
│   ├── generator.py        # Synthetic dataset: random chip geometry + kappa/Q fields
│   └── solver.py           # 3D FDM solver (Conjugate Gradient) for ground truth T
└── training/
    └── loss.py             # PINO residual loss: div(-kappa * grad(T)) + Q
```

## Usage

**Train the model:**
```bash
cd thermal_nn
python main.py
# Trains for 100 epochs on 20 synthetic samples
# Prints loss every 10 epochs
```

**Run the FDM solver standalone:**
```bash
python data/solver.py
# Solves a 16^3 test case and prints T statistics
```

## Data Generation

`generator.py` produces synthetic chip-scale thermal problems:
- A `32³` spatial grid with uniform air conductivity (`kappa = 0.026`)
- A rectangular "chip" region stamped with silicon conductivity (`kappa = 150.0`)
- A localized heat source `Q` centered on the chip
- Random chip placement per sample for geometric diversity

For supervised training, `solver.py` provides a **3D Finite Difference Method (FDM)** solver using `scipy.sparse` + Conjugate Gradient to generate accurate ground truth temperature fields.

## Zero-Shot Generalization

Because TFNO is a resolution-invariant operator, a model trained on `32³` grids can be evaluated on `64³` or `128³` grids at inference time by simply passing higher-resolution inputs — no retraining required.

## Device Support

```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")
```

Automatically uses Apple MPS (M-series Mac) or CUDA GPU.

## Dependencies

```
torch
neuralop
scipy
numpy
```