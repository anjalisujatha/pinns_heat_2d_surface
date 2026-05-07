import torch


def pino_loss(T_pred, kappa, Q, dx=0.1):
    # T_pred shape: [Batch, 1, H, W, D]
    # Calculate gradients using torch.gradient [cite: 27]
    grad = torch.gradient(T_pred.squeeze(1), spacing=dx, edge_order=2)
    dTdx, dTdy, dTdz = grad[0], grad[1], grad[2]

    # Flux = -kappa * grad(T)
    flux_x = -kappa.squeeze(1) * dTdx
    flux_y = -kappa.squeeze(1) * dTdy
    flux_z = -kappa.squeeze(1) * dTdz

    # Divergence of Flux
    div_x = torch.gradient(flux_x, spacing=dx, dim=1)[0]
    div_y = torch.gradient(flux_y, spacing=dx, dim=2)[0]
    div_z = torch.gradient(flux_z, spacing=dx, dim=3)[0]

    # Residual = div(flux) + Q [cite: 28]
    residual = (div_x + div_y + div_z) + Q.squeeze(1)
    return torch.mean(residual ** 2)