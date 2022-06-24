"""Provide functions related to Cryo-BIFE"""
from typing import Callable, Tuple
import numpy as np
import scipy.optimize as so
from scipy.special import logsumexp


class CryoBife:
    """CryoBife provides cryo-bife's prior, likelihood, posterior and
    the optimizer as described in doi: 10.1038/s41598-021-92621-1."""

    @staticmethod
    def integrated_prior(fe_prof: np.ndarray) -> float:
        """Calculate the value of the prior for the given free-energy profile.

        :param fe_prof: Array with the values of the free-energy profile in each
            node of the path.

        :returns: The value of the prior for the free-energy profile.
        """
        acc_der_fe_prof = sum(np.diff(fe_prof) ** 2)
        log_prior = np.log(1 / acc_der_fe_prof**2)

        return log_prior

    @staticmethod
    def gen_img(coord, n_pixels, pixel_size, sigma):

        n_atoms = coord.shape[1]
        norm = 1 / (2 * np.pi * sigma**2 * n_atoms)

        grid_min = -pixel_size * (n_pixels - 1) * 0.5
        grid_max = pixel_size * (n_pixels - 1) * 0.5 + pixel_size

        grid = np.arange(grid_min, grid_max, pixel_size)
        image = np.zeros((n_pixels, n_pixels))

        gauss = np.exp(-0.5 * (((grid[:, None] - coord[0, :]) / sigma) ** 2))[
            :, None
        ] * np.exp(-0.5 * (((grid[:, None] - coord[1, :]) / sigma) ** 2))

        image = gauss.sum(axis=2) * norm

        return image

    @staticmethod
    def likelihood(
        path: np.ndarray, images: np.ndarray, img_params, sigma: float
    ) -> np.ndarray:
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

        n_pixels, pixel_size, sigma_img = img_params

        for i in range(number_of_images):
            for j in range(number_of_nodes):

                coord_rot = np.matmul(images[i]["Q"], path[j])
                img_calc = CryoBife.gen_img(coord_rot, n_pixels, pixel_size, sigma_img)

                prob_matrix[i, j] = -np.sum((images[i]["I"] - img_calc) ** 2) / (
                    2 * sigma**2
                )

        prob_matrix += np.log(norm)
        return prob_matrix

    def energy(
        self,
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
            prior_fxn = self.integrated_prior

        # TODO: Think for a better name for weights
        weights = np.exp(-beta * fe_prof)  # density vec
        weights = weights / np.sum(weights)  # normalize, Eq.(8)

        # Sum here since iid images; logsumexp
        log_likelihood = np.sum(np.log(np.dot(prob_mat, weights)))
        log_prior = kappa * prior_fxn(fe_prof)

        energy = -(log_likelihood + log_prior)

        return energy

    @staticmethod
    def grad_and_energy(
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
            prior_fxn = CryoBife.integrated_prior

        weights = np.exp(-beta * fe_prof)
        weights /= np.sum(weights)

        prob_mat = CryoBife.likelihood(path, images, sigma)

        # Sum here since iid images; logsumexp
        log_likelihood = np.sum(np.log(np.dot(prob_mat, weights)))
        log_prior = kappa * prior_fxn(fe_prof)

        energy = -(log_likelihood + log_prior)

        # Calculate gradient
        gradient = np.zeros_like(path)

        weighted_pmat = (
            weights[:, None] * prob_mat.T / np.sum(prob_mat * weights, axis=1)
        )

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

    def optimizer(
        self,
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

        prob_mat = self.likelihood(path, images, sigma)

        optimized_fe_prof = so.minimize(
            self.energy, initial_fe_prof, method="CG", args=(kappa, prob_mat)
        ).x

        return optimized_fe_prof
