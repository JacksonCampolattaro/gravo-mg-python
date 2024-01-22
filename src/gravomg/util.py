import numpy as np
from scipy import sparse
import gravomg_bindings
from typing import List, Set, Any


def extract_edges(matrix: sparse.coo_matrix):
    return gravomg_bindings.extract_edges(matrix)


def to_homogenous(edge_list: List[Set[Any]]):
    return gravomg_bindings.to_homogenous(edge_list)
