"""Provide functions related to Cryo-BIFE"""
from typing import Callable, Tuple
import numpy as np
import scipy.optimize as so


class CryoBife:
    """CryoBife provides cryo-bife's prior, likelihood, posterior and
    the optimizer as described in doi: 10.1038/s41598-021-92621-1. """

    @staticmethod
    def integrated_prior(fe_prof: np.ndarray) -> float:
        """Calculate the value of the prior for the given free-energy profile.

        :param fe_prof: Array with the values of the free-energy profile in each
            node of the path.

        :returns: The value of the prior for the free-energy profile.
        """
        acc_der_fe_prof = sum(np.diff(fe_prof)**2)
        log_prior = np.log(1 / acc_der_fe_prof**2)

        return log_prior

    @staticmethod
    def likelihood(
            path: np.ndarray,
            images: np.ndarray,
            sigma: float) -> np.ndarray:
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
        prob_matrix = norm * np.exp(-0.5 * 1/sigma**2 *\
                                    np.sum((path[:,None] - images)**2, axis=-1)).T

        return prob_matrix

    def neg_log_posterior(
            self,
            fe_prof: np.ndarray,
            kappa: float,
            prob_mat: np.ndarray,
            beta: float = 1,
            prior_fxn: Callable = None) -> float:
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

        # TODO: Think for a better name for rho
        rho = np.exp(-beta * fe_prof) #density vec
        rho = rho / np.sum(rho) #normalize, Eq.(8)

        # Sum here since iid images; logsumexp
        log_likelihood = np.sum(np.log(np.dot(prob_mat, rho)))
        log_prior = kappa * prior_fxn(fe_prof)

        neg_log_posterior = -(log_likelihood + log_prior)

        return neg_log_posterior

    @staticmethod
    def grad_and_energy(
            path: np.ndarray,
            fe_prof: np.ndarray,
            images: np.ndarray,
            sigma: float,
            kappa: float,
            beta: float = 1,
            prior_fxn: Callable = None) -> Tuple[float, np.ndarray]:
        """Calculate cryo-bife's negative log-posterior.

        :param path: Array with the values of the variables at each node of the path.
                     Shape must be (n_models, n_dimensions).
        :param fe_prof: Array with the values of the free-energy profile (FEP)
                        in each node of the path.
        :param images: Array with all the experimental images.
                       Shape must be (n_images, image_dimensions)
        :param sigma: TODO.
        :param beta: Temperature.
        :param kappa: Scaling factor for the prior.
        :prior_fxn: Function used to calculate the FEP's prior

        :returns: Value of the negative log-posterior
        """
        if prior_fxn is None:
            # Default is the prior from the paper
            prior_fxn = CryoBife.integrated_prior

        # TODO: Think for a better name for rho
        rho = np.exp(-beta * fe_prof) #density vec
        rho = rho / np.sum(rho) #normalize, Eq.(8)

        prob_mat = CryoBife.likelihood(path, images, sigma)

        # Sum here since iid images; logsumexp
        log_likelihood = np.sum(np.log(np.dot(prob_mat, rho)))
        log_prior = kappa * prior_fxn(fe_prof)

        neg_log_posterior = -(log_likelihood + log_prior)

        # Calculate gradient
        grad = np.zeros_like(path)

        weighted_pmat = rho[:,None] * prob_mat.T / np.sum(prob_mat * rho, axis=1)

        grad[:,0] = -1 / sigma**2 * np.sum((images[:,0] - path[:,0][:,None]) *\
                    weighted_pmat, axis=1)
        grad[:,1] = -1 / sigma**2 * np.sum((images[:,1] - path[:,1][:,None]) *\
                    weighted_pmat, axis=1)

        return neg_log_posterior, grad


    def optimizer(
            self,
            path: np.ndarray,
            images: np.ndarray,
            sigma: float,
            initial_fe_prof: np.ndarray = None) -> np.ndarray:
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
        n_models = path.shape[0]

        if initial_fe_prof is None:

            initial_fe_prof = 1.0 * np.random.randn(n_models)

        prob_mat = self.likelihood(path, images, sigma)

        optimized_fe_prof = so.minimize(self.neg_log_posterior,
                                        initial_fe_prof,
                                        method='CG',
                                        args=(kappa, prob_mat))

        return optimized_fe_prof.x


class CryoVIFE(object):
    def __init__(self, images, sigma, beta):
        super().__init__()
        self.images = images
        self.sigma = sigma
        self.beta = beta

    def __call__(self, path, fe_prof):
        return CryoVIFE.grad_and_energy(path, fe_prof, self.images, 
                                        self.sigma, self.beta)

    @staticmethod
    def grad_and_energy(
            path: np.ndarray,
            fe_prof: np.ndarray,
            images: np.ndarray,
            sigma: float,
            beta: float = 1) -> Tuple[float, np.ndarray]:
        """Calculate loss function for VI. 

        :param path: Array with the values of the variables at each node of the path.
                     Shape must be (n_models, n_dimensions).
        :param fe_prof: Array with the values of the free-energy profile (FEP)
                        in each node of the path.
        :param images: Array with all the experimental images.
                       Shape must be (n_images, image_dimensions)
        :param sigma: TODO.
        :param beta: Temperature.
        :prior_fxn: Function used to calculate the FEP's prior

        :returns: Value of the negative log-posterior
        """
        num_nodes = path.shape[0]

        # TODO: Think for a better name for rho
        rho = np.exp(-beta * fe_prof) #density vec
        rho = rho / np.sum(rho) #normalize, Eq.(8)

        
        path_image_dists = (path - images[:, None])
        variance= sigma**2
        lognorm = -np.log(2 * np.pi * variance)
        log_prob_mat = -np.sum(path_image_dists**2, axis=-1) / (2 * variance)  # Unnormalized, we can add the normalization constant later.

        # Sum here since iid images
        variational_cost = - np.mean(np.dot(log_prob_mat, rho)) + num_nodes * lognorm
        entropy = np.dot(rho, np.log(rho))
        negative_elbo = variational_cost + entropy

        # Calculate gradient with respect to nodes 
        grad = np.mean(path_image_dists, axis=0) * rho.reshape(-1, 1) / (variance)

        return negative_elbo, grad