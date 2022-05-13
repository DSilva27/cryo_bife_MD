"""Provide stochastic gradient descent optimizator for path optimization with cryo-bife"""
from typing import Callable, Tuple
import numpy as np
from cryo_bimep.utils import prep_for_mpi


def run_stochastic_gd(
    initial_path: np.ndarray,
    fe_prof: np.ndarray,
    grad_and_energy_func: Callable,
    grad_and_energy_args: Tuple,
    mpi_params,
    images: np.ndarray,
    steps: float,
    step_size: float = 0.0001,
    batch_size: int = None,
) -> np.ndarray:
    """Run simulation using stochastic gradient descent.

    Parameters
    ----------
    initial_path: np.ndarray
        Array with the initial values of the free-energy profile in each
        node of the path.
    fe_prof: np.ndarray
        Array with the values of the free-energy profile (FEP)
        in each node of the path.
    grad_and_energy_func: Callable
        Function that returns the energy and gradient.
        It must take as arguments (path, fe_profile, *args)
    grad_and_energy_args: Tuple
        extra arguments for grad_and_energy_func
    steps: int
        Number of gradient descent steps
    step_size: float
        Step size for new proposals
    batch_size: int
        Batch size for the images to use

    :returns: Last accepted path
    """

    # Set up MPI stuff
    rank, world_size, comm = mpi_params

    if batch_size is None:
        batch_size = images.shape[0]

    number_of_batches = images.shape[0] // batch_size
    residual_batches = images.shape[0] % batch_size

    sim_path = initial_path.copy()

    path_rank = prep_for_mpi(sim_path, rank, world_size)
    images_shuffled = images.copy()

    for _ in range(steps):

        if rank == 0:
            images_shuffled = images.copy()
            np.random.shuffle(images_shuffled)

        comm.bcast(images_shuffled, root=0)

        for i in range(number_of_batches):

            images_batch = images_shuffled[i * batch_size : (i + 1) * batch_size]

            comm.Barrier()
            __, grad = grad_and_energy_func(path_rank, fe_prof, images_batch, *grad_and_energy_args)
            path_rank += -step_size * grad

        if residual_batches != 0:

            images_batch = images_shuffled[(number_of_batches - 1) * batch_size :]

            comm.Barrier()
            __, grad = grad_and_energy_func(path_rank, fe_prof, images_batch, *grad_and_energy_args)
            path_rank += -step_size * grad

    lenghts = np.array(comm.allgather(path_rank.size))
    tmp_path = np.empty((initial_path.size))

    comm.Allgatherv(path_rank, (tmp_path, lenghts))
    sim_path = tmp_path.reshape(initial_path.shape)

    # returns last accepted path
    return sim_path
