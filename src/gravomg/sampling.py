import numpy

import gravomg_bindings


def fast_disc_sample(pos: numpy.ndarray, edges: numpy.ndarray, radius: float):
    return gravomg_bindings.fast_disc_sample(pos, edges, radius)


def fast_disc_sample_by_approximate_ratio(pos: numpy.ndarray, edges: numpy.ndarray, ratio: float):
    radius = gravomg_bindings.average_edge_length(pos, edges) * (ratio ** (1.0 / 3))
    print(gravomg_bindings.average_edge_length(pos, edges), radius)
    return gravomg_bindings.fast_disc_sample(pos, edges, radius)
