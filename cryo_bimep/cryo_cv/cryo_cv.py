"""Provide functions related to Cryo-CV"""
from typing import Callable, Tuple
import numpy as np
import scipy.optimize as so


def _calc_likelihood(
    path_samples: np.ndarray, images: np.ndarray, sigma: float
) -> np.ndarray:

    norm = 1 / (2 * np.pi * sigma**2)

    prob_matrix = (
        np.exp(
            -0.5
            / sigma**2
            * (np.sum((path_samples[:, :, None] - images) ** 2, axis=-1))
        )
        * norm
    )

    return prob_matrix


def _calc_cv(configuration):

    """
    I don't really do anything in the toy model, but I will have to exist
    when we move to real systems!!!!
    """

    return configuration.copy()


def _calc_energy_and_grad_non_vectorized(path, path_samples, fe_prof, images):

    path_cvs = _calc_cv(path)
    traj_cvs = _calc_cv(path_samples)

    gradient = np.zeros_like(path_cvs)

    diff_cvs = traj_cvs - path_cvs[:, np.newaxis, :]

    # !TODO set up cryo-bf sigma
    prob_mat = _calc_likelihood(path_samples, images, 1)

    nu = 1
    weights = np.exp(-fe_prof)

    for i in range(path_cvs.shape[0]):  # sobre los nodos
        for j in range(path_cvs.shape[1]):  # sobre las dimensiones (0, 1)
            for k in range(images.shape[0]):  # sobre las imágenes

                mean1 = np.mean(diff_cvs[i, :, j] * prob_mat[i, :, k])
                mean2 = np.mean(prob_mat[i, :, k])
                mean3 = np.mean(diff_cvs[i, :, j])

                denominator = np.sum(weights * np.mean(prob_mat, axis=1)[:, k])

                gradient[i, j] += (
                    2 * nu * weights[i] * (mean1 - mean2 * mean3) / denominator
                )

    energy = 0.0
    for i in range(images.shape[0]):
        tmp_sum = 0.0
        for j in range(path.shape[0]):

            mean1 = np.mean(prob_mat[j, :, i])
            denominator = np.sum(weights)

            tmp_sum += mean1 * weights[j] / denominator

        energy += np.log(tmp_sum)

    return energy, gradient


def calc_energy_and_grad(path, trajectories, fe_prof, images, spring_constant):

    path_cvs = _calc_cv(path)
    traj_cvs = _calc_cv(trajectories)

    # !TODO set up cryo-bf sigma
    likelihood_matrix = _calc_likelihood(trajectories, images, 1)
    diff_cvs = traj_cvs - path_cvs[:, np.newaxis, :]

    weights = np.exp(-fe_prof)

    # LG-POSTERIOR
    log_posterior = (
        np.mean(likelihood_matrix, axis=1) * weights[:, np.newaxis] / np.sum(weights)
    )
    log_posterior = np.sum(np.log(np.sum(log_posterior, axis=0)))

    # GRADIENT
    exp_value1 = (
        np.einsum("lij, lik -> ljk", diff_cvs, likelihood_matrix)
        / trajectories.shape[1]
    )

    exp_value2 = (
        np.mean(likelihood_matrix, axis=1)[:, np.newaxis, :]
        * np.mean(diff_cvs, axis=1)[:, :, np.newaxis]
    )

    denominator = 1 / np.sum(
        weights[:, np.newaxis] * np.mean(likelihood_matrix, axis=1), axis=0
    )

    gradient = (
        2
        * spring_constant
        * weights[:, np.newaxis]
        * (np.sum((exp_value1 - exp_value2) * denominator, axis=2))
    )

    return -log_posterior, -gradient
