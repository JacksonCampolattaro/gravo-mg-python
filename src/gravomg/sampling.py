import numpy

import gravomg_bindings
import scipy


def fast_disc_sample(pos: numpy.ndarray, edges: scipy.sparse.csr_matrix, radius: float):
    return gravomg_bindings.fast_disc_sample(pos, edges, radius)


def fast_disc_sample_by_approximate_ratio(pos: numpy.ndarray, edges: numpy.ndarray, ratio: float):
    radius = gravomg_bindings.average_edge_length(pos, edges) * (ratio ** (1.0 / 3))
    return gravomg_bindings.fast_disc_sample(pos, edges, radius)
