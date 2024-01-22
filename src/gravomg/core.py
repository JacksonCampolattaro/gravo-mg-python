import numpy

import gravomg_bindings
from gravomg_bindings import Weighting


def construct_prolongation(
        pos: numpy.ndarray, samples:numpy.ndarray, weighting_scheme=Weighting.BARYCENTRIC,
        verbose: bool = False
):
    return gravomg_bindings.construct_prolongation(
        pos, samples,
        weighting_scheme,
        verbose
    )
