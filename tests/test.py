import os
import numpy as np
import gravomg
import torch
import torch_geometric
import torch_cluster
import polyscope
import robust_laplacian

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f'cube.obj')
cube = torch_geometric.io.read_obj(path)
data = torch_geometric.transforms.sample_points.SamplePoints(num=4096, include_normals=True)(cube)
fine_points = data.pos

# Get a neighbor list from the robust laplacian of the point cloud
L, M = robust_laplacian.point_cloud_laplacian(fine_points.numpy())
fine_edges = gravomg.extract_edges(L)
fine_edge_pairs = np.array([(i, j) for i, edge_set in enumerate(fine_edges) for j in edge_set])

# Use fast-disc-sampling to select coarse points
# todo

# samples = gravomg.sampling.fast_disc_sample_by_approximate_ratio(data.pos, neighbor_indices, ratio=0.5)
# samples = torch.LongTensor(samples)
# sampled_positions = torch.gather(
#     data.pos,
#     index=samples.unsqueeze(-1).expand(-1, data.pos.shape[-1]),
#     dim=0
# )

# print(f"selected {sampled_positions.shape[0]}/{data.pos.shape[0]}")

polyscope.init()
polyscope.set_ground_plane_mode('none')
polyscope.register_point_cloud("positions", data.pos, radius=0.0025)
polyscope.register_curve_network("fine edges", data.pos, fine_edge_pairs, radius=0.001)
# polyscope.register_point_cloud("sampled-positions", sampled_positions)
polyscope.show()
