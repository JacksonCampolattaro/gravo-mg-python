import numpy
import gravomg_bindings

from gravomg_bindings import Sampling, Weighting


def construct_prolongation(
        pos: numpy.ndarray, ratio: float = 8.0, nested: bool = False, lowBound: int = 1000,
        sampling_strategy: Sampling = Sampling.FASTDISK, weighting_scheme=Weighting.BARYCENTRIC,
        verbose: bool = False
):
    return gravomg_bindings.construct_prolongation(
        pos, ratio, nested, lowBound,
        sampling_strategy, weighting_scheme,
        verbose
    )
