"""Provide stochastic gradient descent optimizator for path optimization with cryo-bife"""
from typing import Callable, Tuple
import numpy as np

def run_stochastic_gd(
        initial_path: np.ndarray,
        fe_prof: np.ndarray,
        grad_and_energy_func: Callable,
        grad_and_energy_args: Tuple,
        images: np.ndarray,
        steps: float,
        step_size: float = 0.0001,
        batch_size: int = None) -> np.ndarray:
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

    if batch_size is None:
        batch_size = images.shape[0]

    #assert images.shape[0]%batch_size == 0, "Batch size is not a multiple of the number of images"

    sim_path = initial_path.copy()

    #TODO set this up as a parameter for the simulator (fixed_nodes)
    mask = np.ones_like(sim_path)

    mask[0] = np.zeros((2,))
    #mask[7] = np.zeros((2,))
    mask[-1] = np.zeros((2,))

    tol = 1e-3
    old_path = initial_path.copy()

    for _ in range(steps):

        images_batch = images[np.random.randint(images.shape[0], size=batch_size), :]

        __, grad = grad_and_energy_func(sim_path, fe_prof, images_batch, *grad_and_energy_args)
        sim_path += -step_size*grad * mask

        if np.sum(abs(sim_path - old_path)) < tol:
            break

        old_path = sim_path.copy()

    # returns last accepted path
    return sim_path
