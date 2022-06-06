"""Provide Brownian motion simulator for path optimization with cryo-bife"""
from typing import Tuple
import numpy as np
from cryo_bimep.utils import prep_for_mpi


def run_brownian_sim(
    initial_path: np.ndarray,
    mpi_params: Tuple,
    steps: float,
    step_size: float = 0.0001,
) -> np.ndarray:
    """Run Brownian Motion simulation.

    :param initial_path: Array with the initial values of the free-energy profile in each
        node of the path.
    :param fe_prof: Array with the values of the free-energy profile (FEP)
                    in each node of the path.

    :param steps: Number of MALA steps to do
    :step_size: Step size for new proposals

    :returns: Last accepted path
    """
    # Set up MPI stuff
    rank, world_size, comm = mpi_params

    # Calculate "old" variables
    sim_path = initial_path.copy()
    path_rank = prep_for_mpi(sim_path, rank, world_size)

    for _ in range(steps):

        path_rank += np.sqrt(2 * step_size) * np.random.randn(*path_rank.shape)

    lenghts = np.array(comm.allgather(path_rank.size))
    tmp_path = np.empty((initial_path.size))

    comm.Allgatherv(path_rank, (tmp_path, lenghts))
    sim_path = tmp_path.reshape(initial_path.shape)

    return sim_path
