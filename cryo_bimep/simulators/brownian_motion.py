"""Provide Brownian motion simulator for path optimization with cryo-bife"""
import numpy as np


def run_brownian_sim(
    initial_path: np.ndarray,
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

    # Calculate "old" variables
    sim_path = initial_path.copy()
    trajectory = np.zeros((steps, *initial_path.shape))

    for step in range(steps):

        sim_path[1:-1] += np.sqrt(2 * step_size) * np.random.randn(*sim_path[1:-1].shape)

        trajectory[step] = sim_path

    return trajectory
