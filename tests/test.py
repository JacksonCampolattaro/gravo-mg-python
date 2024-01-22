import os
import numpy as np
import gravomg
import torch
import torch_geometric
import torch_cluster
import polyscope
import robust_laplacian

NUM_FINE_POINTS = 4096
REDUCTION_RATIO = 1.0

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f'cube.obj')
cube = torch_geometric.io.read_obj(path)
data = torch_geometric.transforms.sample_points.SamplePoints(num=NUM_FINE_POINTS, include_normals=True)(cube)
fine_points = data.pos

# Get a neighbor list from the robust laplacian of the point cloud
L, M = robust_laplacian.point_cloud_laplacian(fine_points.numpy())
fine_edges = gravomg.extract_edges(L)
fine_edge_pairs = np.array([(i, j) for i, edge_set in enumerate(fine_edges) for j in edge_set])

# Use fast-disc-sampling to select coarse points
radius = (REDUCTION_RATIO ** (1.0 / 3.0)) * gravomg.average_edge_length(fine_points, fine_edges)
coarse_recommendations = gravomg.fast_disc_sample(fine_points, gravomg.to_homogenous(fine_edges), radius)

# Link fine points to their nearest coarse point
parents = gravomg.assign_parents(fine_points, fine_edges, coarse_recommendations)
fine_coarse_pairs = np.array([(i, coarse_recommendations[p]) for i, p in enumerate(parents)])

# Determine coarse edge relationships based on the relationships of their children
coarse_edges = gravomg.extract_coarse_edges(fine_points, fine_edges, coarse_recommendations, parents)
coarse_edge_pairs = np.array([(i, j) for i, edge_set in enumerate(coarse_edges) for j in edge_set])

# Choose positions for the coarse points based on their child positions
coarse_points = gravomg.coarse_from_mean_of_fine_children(fine_points, fine_edges, parents, len(coarse_recommendations))

# Produce voronoi triangles for the coarse points
# (only needed for debugging; this is built into construct_prolongation)
triangles_with_normals, point_triangle_associations = gravomg.construct_voronoi_triangles(coarse_points, coarse_edges)
triangles = [triangle for triangle, _normal in triangles_with_normals]

# Construct prolongation operator
prolongation_matrix = gravomg.construct_prolongation(fine_points, coarse_points, coarse_edges, parents)

# Use the operator to produce projections of the fine points based on their weights
fine_projections = prolongation_matrix.dot(coarse_points)
fine_points_with_projections = np.concatenate([fine_points, fine_projections], axis=0)
fine_projection_pairs = np.array([(i, i + len(fine_points)) for i in range(0, len(fine_points))])

# Display everything with polyscope
polyscope.init()
polyscope.set_ground_plane_mode('none')

# Fine points & edges
polyscope.register_point_cloud("fine points", fine_points, radius=0.0025, enabled=False)
polyscope.register_curve_network("fine edges", fine_points, fine_edge_pairs, radius=0.001, enabled=False)
polyscope.register_curve_network("parent assignments", fine_points, fine_coarse_pairs, radius=0.001, enabled=False)

# Coarse points & edges
polyscope.register_point_cloud("coarse points", coarse_points, radius=0.004, enabled=False)
polyscope.register_curve_network("coarse edges", coarse_points, coarse_edge_pairs, radius=0.0015, enabled=False)

# Voronoi triangulation
polyscope.register_surface_mesh("coarse triangles", coarse_points, triangles, enabled=True)

# Projections
polyscope.register_curve_network(
    "projections",
    fine_points_with_projections, fine_projection_pairs,
    radius=0.0015, enabled=True
)

polyscope.show()
