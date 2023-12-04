import numpy

import gravomg_bindings

def fast_disc_sample(pos: numpy.ndarray, edges: numpy.ndarray, radius: float):
    return gravomg_bindings.fast_disc_sample(pos, edges, radius)
