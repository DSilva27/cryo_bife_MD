"""Provide stochastic gradient descent optimizator for path optimization with cryo-bife"""
from typing import Callable, Tuple
import numpy as np

from cryo_bimep.simulators import string_method

def run_string_stochastic(
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

    number_of_batches = images.shape[0]//batch_size
    residual_batches = images.shape[0]%batch_size

    sim_path = initial_path.copy()
    sim_path, Norm_tan_vec = string_method.run_string_method(sim_path)

    #TODO set this up as a parameter for the simulator (fixed_nodes)
    mask = np.ones_like(sim_path)

    mask[0] = np.zeros((2,))
    #mask[7] = np.zeros((2,))
    mask[-1] = np.zeros((2,))

    tol = 1e-3
    old_path = initial_path.copy()

    for _ in range(steps):

        images_shuffled = images.copy()
        np.random.shuffle(images_shuffled)

        for i in range(number_of_batches):

            images_batch = images_shuffled[i*batch_size:(i+1)*batch_size]

            __, grad = grad_and_energy_func(sim_path, fe_prof, images_batch, *grad_and_energy_args)
            #sim_path += -step_size*grad * mask

            Perp_grad = [np.dot(Norm_tan_vec.T[0,:],grad[:,0]),np.dot(Norm_tan_vec.T[1,:],grad[:,1])]*Norm_tan_vec
            #sim_path += step_size*(grad * mask)# - Perp_grad * mask)
            sim_path += -step_size*(grad * mask - Perp_grad * mask)

        if residual_batches != 0:

            images_batch = images_shuffled[(number_of_batches-1)*batch_size:]
            __, grad = grad_and_energy_func(sim_path, fe_prof, images_batch, *grad_and_energy_args)
            #sim_path += -step_size*grad * mask

            Perp_grad = [np.dot(Norm_tan_vec.T[0,:],grad[:,0]),np.dot(Norm_tan_vec.T[1,:],grad[:,1])]*Norm_tan_vec
            #sim_path += step_size*(grad * mask)# - Perp_grad * mask)
            sim_path += -step_size*(grad * mask - Perp_grad * mask)

        if np.sum(abs(sim_path - old_path)) < tol:
            break

        sim_path, Norm_tan_vec = string_method.run_string_method(sim_path)
        old_path = sim_path.copy()

    # returns last accepted path
    return sim_path
