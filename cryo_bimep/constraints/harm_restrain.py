"Provides constraint on harmonic restraint for a restrained MD-like simulation"
from typing import Tuple
import numpy as np


def harm_rest_energy_and_grad(path: np.ndarray, ref_path: np.array, kappa: float) -> Tuple[float, np.ndarray]:
    """Calculate harmonic distance constraint energy and grad for a path when compared to a reference path.

    :param path: Array with the initial values of the free-energy profile in each
                 node of the path.
    :kappa_2: harmonic constant for the constraint
    :equib_dist: equilibrium distance between nodes of the path

    :returns: harmonic distance constraint energy and gradient
    """

    energy = 0.5 * kappa * np.sum(np.sum((path - ref_path) ** 2, axis=1))
    gradient = kappa * (path - ref_path)

    return energy, gradient
