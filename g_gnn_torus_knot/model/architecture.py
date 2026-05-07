import torch
import torch_geometric
from e3nn import o3
from e3nn.nn.models.gate_points_2101 import Network

# --- 1. Model Configuration ---
# Defining irreps based on your physical variables
input_irreps = "1x0e + 1x1o"  # Scalar (mass/params) + Vector (Position)
output_irreps = "1x1o"        # Vector (Velocity/Next state)

# Standard E(3) Equivariant Network
model = Network(
    irreps_in=input_irreps,
    irreps_hidden="32x0e + 16x1o + 8x2e", # Captures scalars, vectors, and tensors
    irreps_out=output_irreps,
    irreps_node_attr="1x0e",            # Potential to pass (p, q) as node attributes
    irreps_edge_attr="1x1o",              # Spherical harmonics of edge direction vectors
    layers=3,
    max_radius=2.0,                     # Search radius for "neighborhood" on the knot
    number_of_basis=10,
    radial_layers=1,
    radial_neurons=64,
    num_neighbors=5.0,
    num_nodes=100.0,
    reduce_output=False,
)

print("Equivariant GNN for Torus Knot dynamics initialized.")
