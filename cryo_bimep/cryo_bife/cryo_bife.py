"""Provide functions related to Cryo-BIFE"""
from typing import Callable, Tuple
import numpy as np
import scipy.optimize as so


def _integrated_prior(fe_prof: np.ndarray) -> float:
    """Calculate the value of the prior for the given free-energy profile.

    :param fe_prof: Array with the values of the free-energy profile in each
        node of the path.

    :returns: The value of the prior for the free-energy profile.
    """
    acc_der_fe_prof = sum(np.diff(fe_prof) ** 2)
    log_prior = np.log(1 / acc_der_fe_prof**2)

    return log_prior


def calc_likelihood(path: np.ndarray, images: np.ndarray, sigma: float) -> np.ndarray:
    """Calculate cryo-bife's likelihood matrix given a path and a dataset of images
    :param path: Array with the values of the variables at each node of the path.
                    Shape must be (n_models, n_dimensions).
    :param images: Array with all the experimental images.
                    Shape must be (n_images, image_dimensions)
    :param sigma: Overall noise among the images
    :returns: Array with the likelihood of observing each image given each model.
                Shape will be (n_images, n_models)
    """

    number_of_nodes = path.shape[0]
    number_of_images = images.shape[0]
    prob_matrix = np.zeros((number_of_images, number_of_nodes))

    norm = 1 / (2 * np.pi * sigma**2)
    prob_matrix = (
        norm
        * np.exp(
            -0.5 * 1 / sigma**2 * np.sum((path[:, None] - images) ** 2, axis=-1)
        ).T
    )

    return prob_matrix


def calc_energy(
    fe_prof: np.ndarray,
    kappa: float,
    prob_mat: np.ndarray,
    beta: float = 1,
    prior_fxn: Callable = None,
) -> float:
    """Calculate cryo-bife's negative log-posterior.
    :param fe_prof: Array with the values of the free-energy profile (FEP)
                    in each node of the path.
    :param beta: Temperature.
    :param kappa: Scaling factor for the prior.
    :prior_fxn: Function used to calculate the FEP's prior
    :returns: Value of the negative log-posterior
    """

    if prior_fxn is None:
        # Default is the prior from the paper
        prior_fxn = _integrated_prior

    # TODO: Think for a better name for weights
    weights = np.exp(-beta * fe_prof)  # density vec
    weights = weights / np.sum(weights)  # normalize, Eq.(8)

    # Sum here since iid images; logsumexp
    log_likelihood = np.sum(np.log(np.dot(prob_mat, weights)))
    log_prior = kappa * prior_fxn(fe_prof)

    energy = -(log_likelihood + log_prior)

    return energy


def calc_grad_and_energy(
    path: np.ndarray,
    fe_prof: np.ndarray,
    images: np.ndarray,
    sigma: float,
    kappa: float,
    beta: float = 1,
    prior_fxn: Callable = None,
) -> Tuple[float, np.ndarray]:
    """Calculate cryo-bife's negative log-posterior.
    :param path: Array with the values of the variables at each node of the path.
                    Shape must be (n_models, n_dimensions).
    :param fe_prof: Array with the values of the free-energy profile (FEP)
                    in each node of the path.
    :param images: Array with all the experimental images.
                    Shape must be (n_images, image_dimensions)
    :param sigma: TODO.
    :param kappa: Scaling factor for the prior.
    :param mpi_params: rank, world_size and communicator
    :param beta: Inverse temperature.
    :prior_fxn: Function used to calculate the FEP's prior
    :returns: Value of the negative log-posterior
    """

    if prior_fxn is None:
        # Default is the prior from the paper
        prior_fxn = _integrated_prior

    weights = np.exp(-beta * fe_prof)
    weights /= np.sum(weights)

    prob_mat = calc_likelihood(path, images, sigma)

    # Sum here since iid images; logsumexp
    log_likelihood = np.sum(np.log(np.dot(prob_mat, weights)))
    log_prior = kappa * prior_fxn(fe_prof)

    energy = -(log_likelihood + log_prior)

    # Calculate gradient
    gradient = np.zeros_like(path)

    weighted_pmat = weights[:, None] * prob_mat.T / np.sum(prob_mat * weights, axis=1)

    gradient[:, 0] = (
        -1
        / sigma**2
        * np.sum((images[:, 0] - path[:, 0][:, None]) * weighted_pmat, axis=1)
    )
    gradient[:, 1] = (
        -1
        / sigma**2
        * np.sum((images[:, 1] - path[:, 1][:, None]) * weighted_pmat, axis=1)
    )

    gradient[0] *= 0.0
    gradient[-1] *= 0.0

    return energy, gradient


def optimize_free_energy(
    path: np.ndarray,
    images: np.ndarray,
    sigma: float,
    initial_fe_prof: np.ndarray = None,
) -> np.ndarray:
    """Find the optimal free-energy profile given a path and a dataset of images

    :param path: Array with the values of the variables at each node of the path.
                    Shape must be (n_models, n_dimensions).
    :param images: Array with all the experimental images.
                    Shape must be (n_images, image_dimensions)
    :param sigma: TODO.
    :param fe_prof: Initial guess for the free-energy profile

    :returns: Optimized free-energy profile
    """

    kappa = 1

    if initial_fe_prof is None:

        initial_fe_prof = 1.0 * np.random.randn(path.shape[0])

    prob_mat = calc_likelihood(path, images, sigma)

    optimized_fe_prof = so.minimize(
        calc_energy, initial_fe_prof, method="CG", args=(kappa, prob_mat)
    ).x

    return optimized_fe_prof
