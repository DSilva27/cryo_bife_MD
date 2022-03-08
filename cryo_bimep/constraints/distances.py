"Provides constraint on distances between nodes of the path"
from typing import Tuple
import numpy as np

def dist_energy_and_grad(
        path: np.ndarray,
        kappa_2: float,
        equib_dist: float) -> Tuple[float, np.ndarray]:
    """Calculate harmonic distance constraint energy and grad for a path.

    :param path: Array with the initial values of the free-energy profile in each
                 node of the path.
    :kappa_2: harmonic constant for the constraint
    :equib_dist: equilibrium distance between nodes of the path

    :returns: harmonic distance constraint energy and gradient
    """

    # Calculate energy
    energy_dist = np.sum((np.sqrt(np.sum((path[:-1] - path[1:])**2, axis=1)) - equib_dist)**2)
    energy_dist *= 0.5 * kappa_2

    # Calculate distances between two or more nodes
    def distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2, axis=-1))

    # Calculate gradient
    grad_dist = np.zeros_like(path)

    grad_dist[0] = (1 - equib_dist / distance(path[0], path[1])) *\
                   (path[0] - path[1])

    grad_dist[-1] = (1 - equib_dist / distance(path[-1], path[-2])) *\
                    (path[-1] - path[-2])

    grad_dist[1:-1] = (1 - equib_dist / distance(path[1:-1], path[2:])[:,None]) *\
                      (path[1:-1] - path[2:]) +\
                      (1 - equib_dist / distance(path[1:-1], path[:-2])[:,None]) *\
                      (path[1:-1] - path[:-2])

    grad_dist *= kappa_2

    return energy_dist, grad_dist
