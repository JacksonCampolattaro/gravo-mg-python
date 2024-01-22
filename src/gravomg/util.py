import numpy as np
from scipy import sparse
import gravomg_bindings


def extract_edges(matrix: sparse.coo_matrix):
    return gravomg_bindings.extract_edges(matrix)