"""Provide Langevin simulator for path optimization with cryo-bife"""
from typing import Callable, Tuple
import numpy as np


def run_euler_maruyama(
    initial_path: np.ndarray,
    grad_and_energy_func: Callable,
    grad_and_energy_args: Tuple,
    steps: float,
    step_size: float = 0.0001,
    stride=None,
) -> np.ndarray:
    """Run simulation using Euler-Maruyama algorithm.
    :param initial_path: Array with the initial values of the free-energy profile in each
        node of the path.
    :param grad_and_energy_func: Function that returns the energy and gradient.
    :param grad_and_energy_args: extra arguments for grad_and_energy_func
    :param steps: Number of MALA steps to do
    :step_size: Step size for new proposals
    :returns: Last accepted path
    """

    # Calculate "old" variables
    sim_path = initial_path.copy()

    if stride is None:
        stride = steps

    if steps % stride == 0:
        trajectory = np.zeros((steps // stride, *initial_path.shape))

    else:
        trajectory = np.zeros((steps // stride + 1, *initial_path.shape))

    counter = 0
    for step in range(steps):

        # Calculate new proposal
        __, grad = grad_and_energy_func(sim_path, *grad_and_energy_args)
        sim_path[1:-1] += (
            -step_size * grad + np.sqrt(2 * step_size) * np.random.randn(*grad.shape)
        )[1:-1]

        if (step + 1) % stride == 0:
            trajectory[counter] = sim_path
            counter += 1

    if steps % stride != 0:
        trajectory[-1] = sim_path

    # returns last accepted path
    return trajectory
