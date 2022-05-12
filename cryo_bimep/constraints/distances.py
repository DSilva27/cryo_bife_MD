"Provides constraint on distances between nodes of the path"
from typing import Tuple
import numpy as np
from cryo_bimep.utils import prep_for_mpi


def dist_energy_and_grad(
        path_rank: np.ndarray,
        kappa_2: float,
        equib_dist: float,
        mpi_params) -> Tuple[float, np.ndarray]:
    """Calculate harmonic distance constraint energy and grad for a path.

    :param path: Array with the initial values of the free-energy profile in each
                 node of the path.
    :kappa_2: harmonic constant for the constraint
    :equib_dist: equilibrium distance between nodes of the path

    :returns: harmonic distance constraint energy and gradient
    """

    # Calculate distances between two or more nodes
    def distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2, axis=-1))

    rank, world_size, comm = mpi_params

    if world_size == 1:
        n_nodes = path_rank.shape[0]

    else:
        n_nodes = world_size + 2

    lenghts = np.array(comm.allgather(path_rank.size))
    tmp_path = np.empty((n_nodes * path_rank.shape[1]))

    comm.Gatherv(path_rank, (tmp_path, lenghts), root=0)

    if rank == 0:

        path = tmp_path.reshape(n_nodes, path_rank.shape[1])

        # Calculate energy
        energy_dist = np.sum((np.sqrt(np.sum((path[:-1] - path[1:])**2, axis=1)) - equib_dist)**2)
        energy_dist *= 0.5 * kappa_2

        # Calculate gradient
        grad_dist = np.zeros_like(path)

        grad_dist[1:-1] = (1 - equib_dist / distance(path[1:-1], path[2:])[:,None]) *\
                          (path[1:-1] - path[2:]) +\
                          (1 - equib_dist / distance(path[1:-1], path[:-2])[:,None]) *\
                          (path[1:-1] - path[:-2])

        grad_dist *= kappa_2

    else:
        energy_dist = 0.0
        grad_dist = np.empty((n_nodes, path_rank.shape[1]))

    comm.bcast(energy_dist, root=0)
    comm.bcast(grad_dist, root=0)

    grad_dist = prep_for_mpi(grad_dist, rank, world_size)

    return energy_dist, grad_dist
