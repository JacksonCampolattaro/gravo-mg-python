import numpy
import numpy as np
from scipy import sparse
import gravomg_bindings
from typing import List, Set, Any


def to_edge_distance_matrix(matrix: sparse.coo_matrix, points: numpy.ndarray):
    return gravomg_bindings.to_edge_distance_matrix(matrix, points)

def extract_edges(matrix: sparse.coo_matrix):
    return gravomg_bindings.extract_edges(matrix)
