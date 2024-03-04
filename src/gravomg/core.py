import numpy
import scipy
from typing import List, Set, Any

import gravomg_bindings
from gravomg_bindings import Weighting


def assign_parents(
        fine_points: numpy.ndarray,
        fine_neighbors: scipy.sparse.csr_matrix,
        coarse_recommendations: List[int]
) -> List[int]:
    return gravomg_bindings.assign_parents(fine_points, fine_neighbors, coarse_recommendations)


def average_edge_length(pos: numpy.ndarray, edges: numpy.ndarray) -> float:
    return gravomg_bindings.average_edge_length(pos, edges)


def extract_coarse_edges(
        fine_points: numpy.ndarray,
        fine_neighbors: scipy.sparse.csr_matrix,
        coarse_recommendations: List[int],
        fine_parents: List[int]
) -> List[Set[int]]:
    return gravomg_bindings.extract_coarse_edges(fine_points, fine_neighbors, coarse_recommendations, fine_parents)


def coarse_from_mean_of_fine_children(
        fine_points: numpy.ndarray,
        fine_neighbors: scipy.sparse.csr_matrix,
        fine_parents: List[int],
        num_coarse_points: int  # todo: not strictly necessary!
) -> numpy.ndarray:
    return gravomg_bindings.coarse_from_mean_of_fine_children(
        fine_points,
        fine_neighbors,
        fine_parents,
        num_coarse_points
    )


def construct_voronoi_triangles(
        coarse_points: numpy.ndarray,
        coarse_edges: scipy.sparse.csr_matrix
):
    return gravomg_bindings.construct_voronoi_triangles(coarse_points, coarse_edges)


def construct_prolongation(
        fine_points: numpy.ndarray,
        coarse_points: numpy.ndarray,
        coarse_edges: scipy.sparse.csr_matrix,
        parents: List[int],
        weighting_scheme=Weighting.BARYCENTRIC
):
    return gravomg_bindings.construct_prolongation(
        fine_points,
        coarse_points,
        coarse_edges,
        parents,
        weighting_scheme
    )
