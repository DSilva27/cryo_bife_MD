"Provides constraint on harmonic restraint for a restrained MD-like simulation"
from typing import Tuple
import numpy as np
from cryo_bimep.utils import prep_for_mpi


def harm_rest_energy_and_grad(
    path_rank: np.ndarray, ref_path: np.array, kappa: float, mpi_params
) -> Tuple[float, np.ndarray]:
    """Calculate harmonic distance constraint energy and grad for a path when compared to a reference path.

    :param path: Array with the initial values of the free-energy profile in each
                 node of the path.
    :kappa_2: harmonic constant for the constraint
    :equib_dist: equilibrium distance between nodes of the path

    :returns: harmonic distance constraint energy and gradient
    """

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
        energy = 0.5 * kappa * np.sum(np.sum((path - ref_path)**2, axis=1))

    else:
        energy = 0.0

    comm.bcast(energy, root=0)

    ref_path_rank = prep_for_mpi(ref_path, rank, world_size)
    gradient = kappa * (path_rank - ref_path_rank)

    return energy, gradient
