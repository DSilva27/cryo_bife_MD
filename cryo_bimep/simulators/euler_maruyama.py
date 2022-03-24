"""Provide MALA simulator for path optimization with cryo-bife"""
from typing import Callable, Tuple
import numpy as np


def run_euler_maruyama(
        initial_path: np.ndarray,
        fe_prof: np.ndarray,
        grad_and_energy_func: Callable,
        grad_and_energy_args: Tuple,
        steps: float,
        step_size: float = 0.0001) -> np.ndarray:
    """Run simulation using MALA algorithm.

    :param initial_path: Array with the initial values of the free-energy profile in each
        node of the path.
    :param fe_prof: Array with the values of the free-energy profile (FEP)
                    in each node of the path.
    :param grad_and_energy_func: Function that returns the energy and gradient.
                                 It must take as arguments (path, fe_profile, *args)
    :param grad_and_energy_args: extra arguments for grad_and_energy_func
    :param steps: Number of MALA steps to do
    :step_size: Step size for new proposals

    :returns: Last accepted path
    """

    # Calculate "old" variables
    old_path = initial_path.copy()
    old_energy, old_grad = grad_and_energy_func(old_path, fe_prof, *grad_and_energy_args)

    for _ in range(steps):

        new_path = old_path.copy()

        # Selecting which replica to update
        path_index = [np.random.randint(0, initial_path.shape[0]), np.random.randint(0, 2)]
        while path_index[0] in (0, 7, 13):
            path_index[0] = np.random.randint(0, initial_path.shape[0])

        # Select which coordinate to update
        path_index = tuple(path_index)

        # Calculate new proposal
        old_path[path_index] += -step_size*old_grad[path_index] +\
                                              np.sqrt(2*step_size) * np.random.randn()
        #new_energy, new_grad = grad_and_energy_func(new_path, fe_prof, *grad_and_energy_args)


    # returns last accepted path
    return old_path