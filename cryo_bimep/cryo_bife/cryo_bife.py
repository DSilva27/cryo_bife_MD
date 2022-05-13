"""Provide functions related to Cryo-BIFE"""
from typing import Callable, Tuple
import numpy as np
import scipy.optimize as so
from scipy.special import logsumexp
from cryo_bimep.utils import prep_for_mpi


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

        gauss = np.exp(-0.5 * (((grid[:, None] - coord[0, :]) / sigma)**2))[:, None] *\
                np.exp(-0.5 * (((grid[:, None] - coord[1, :]) / sigma)**2))

        image = gauss.sum(axis=2) * norm

        return image

    @staticmethod
    def likelihood(
            path: np.ndarray,
            images: np.ndarray,
            img_params,
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

        n_pixels, pixel_size, sigma_img = img_params

        for i in range(number_of_images):
            for j in range(number_of_nodes):
                
                coord_rot = np.matmul(images[i]["Q"], path[j])
                img_calc = CryoBife.gen_img(coord_rot, n_pixels, pixel_size, sigma_img)

                prob_matrix[i, j] = -np.sum((images[i]["I"] - img_calc)**2) / (2 * sigma**2)

        prob_matrix += np.log(norm)
        return prob_matrix

    def neg_log_posterior(
        self, fe_prof: np.ndarray, kappa: float, prob_mat: np.ndarray, beta: float = 1, prior_fxn: Callable = None
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

        neg_log_posterior = -(log_likelihood + log_prior)

        return neg_log_posterior

    @staticmethod
    def grad_and_energy(
        path: np.ndarray,
        fe_prof: np.ndarray,
        images: np.ndarray,
        img_params: Tuple,
        sigma: float,
        kappa: float,
        mpi_params: Tuple,
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

        # Setting up parallel stuff
        rank, world_size, comm = mpi_params

        # Setting up numbers of things
        n_images = images.shape[0]
        n_atoms = path.shape[2]
        if world_size == 1:
            n_nodes = path.shape[0]

        else:
            n_nodes = world_size + 2

        # Calculate constants 
        n_pixels, pixel_size, sigma_img = img_params
        norm_grad = 1 / (2 * np.pi * sigma_img**4 * sigma**2 * n_atoms)

        grid_min = -pixel_size * (n_pixels - 1) * 0.5
        grid_max = pixel_size * (n_pixels - 1) * 0.5 + pixel_size
        grid = np.arange(grid_min, grid_max, pixel_size)
        
        if prior_fxn is None:
            # Default is the prior from the paper
            prior_fxn = CryoBife.integrated_prior

        weights = -beta * fe_prof
        inv_total_weight = np.sum(np.exp(weights))

        prob_mat_rank = CryoBife.likelihood(path, images, img_params, sigma)
        lenghts = np.array(comm.allgather(prob_mat_rank.size))

        prob_mat = np.empty((n_images * n_nodes))
        comm.Allgatherv(prob_mat_rank.T.flatten(), (prob_mat, lenghts))
        prob_mat = prob_mat.reshape(n_nodes, n_images).T

        # Sum here since iid images; logsumexp
        log_likelihood = logsumexp(prob_mat + weights, b=inv_total_weight, axis=1).sum()
        log_prior = kappa * prior_fxn(fe_prof)
        cryo_bife_energy = -(log_likelihood + log_prior)

        # Calculate gradient
        grad = np.zeros_like(path)
        weights = np.exp(weights) * inv_total_weight
        prob_mat = np.exp(prob_mat)
        weighted_pmat = 1 / np.einsum("ij,j->i", prob_mat, weights)

        grad_tmp = np.zeros_like(path[0])

        if world_size == 1:

            for i in range(1, n_nodes - 1):

                gaussians = np.exp(-0.5 * (((grid[:, None] - path[i][0, :])**2 / sigma_img**2)[:, None] + (grid[:, None] - path[i][1, :])**2 / sigma_img**2))

                der_gaussians_x = (grid[:, None] - path[i][0, :])
                der_gaussians_y = (grid[:, None] - path[i][1, :])
                img_node = CryoBife.gen_img(path[i], *img_params)

                for j in range(n_images):

                    #node_rot = np.matmul(images[j]["Q"], path[i])

                    dif_images = images[j]["I"] - img_node

                    grad_tmp[0] = np.einsum("ij, ik, ijk->j", dif_images, der_gaussians_x, gaussians)
                    grad_tmp[1] = np.einsum("ij, jk, ijk->j", dif_images, der_gaussians_y, gaussians)

                    grad_tmp[0] *= norm_grad * prob_mat[j, i]
                    grad_tmp[1] *= norm_grad * prob_mat[j, i]

                    #grad_tmp = np.matmul(images[j]["Q_inv"], grad_tmp)

                    grad[i, :] += grad_tmp * weights[i] * weighted_pmat[j]

        else:
            if rank == 0:

                gaussians = np.exp(-0.5 * (((grid[:, None] - path[1][0, :])**2 / sigma_img**2)[:, None] + (grid[:, None] - path[1][1, :])**2 / sigma_img**2))

                der_gaussians_x = (grid[:, None] - path[1][0, :])
                der_gaussians_y = (grid[:, None] - path[1][1, :])
                img_node = CryoBife.gen_img(path[1], *img_params)

                for j in range(n_images):

                    #node_rot = np.matmul(images[j]["Q"], path[i])

                    dif_images = images[j]["I"] - img_node

                    grad_tmp[0] = np.einsum("ij, ik, ijk->j", dif_images, der_gaussians_x, gaussians)
                    grad_tmp[1] = np.einsum("ij, jk, ijk->j", dif_images, der_gaussians_y, gaussians)

                    grad_tmp[0] *= norm_grad * prob_mat[j, 1]
                    grad_tmp[1] *= norm_grad * prob_mat[j, 1]

                    #grad_tmp = np.matmul(images[j]["Q_inv"], grad_tmp)

                    grad[1, :] += grad_tmp * weights[1] * weighted_pmat[j]

            else:

                gaussians = np.exp(-0.5 * (((grid[:, None] - path[0][0, :])**2 / sigma_img**2)[:, None] + (grid[:, None] - path[0][1, :])**2 / sigma_img**2))

                der_gaussians_x = (grid[:, None] - path[0][0, :])
                der_gaussians_y = (grid[:, None] - path[0][1, :])
                img_node = CryoBife.gen_img(path[0], *img_params)

                for j in range(n_images):

                    #node_rot = np.matmul(images[j]["Q"], path[i])

                    dif_images = images[j]["I"] - img_node

                    grad_tmp[0] = np.einsum("ij, ik, ijk->j", dif_images, der_gaussians_x, gaussians)
                    grad_tmp[1] = np.einsum("ij, jk, ijk->j", dif_images, der_gaussians_y, gaussians)

                    grad_tmp[0] *= norm_grad * prob_mat[j, 1]
                    grad_tmp[1] *= norm_grad * prob_mat[j, 1]

                    #grad_tmp = np.matmul(images[j]["Q_inv"], grad_tmp)

                    grad[0, :] += grad_tmp * weights[rank + 1] * weighted_pmat[j]

        return cryo_bife_energy, grad


    def optimizer(
        self, path: np.ndarray, images: np.ndarray, sigma: float, mpi_params: Tuple, initial_fe_prof: np.ndarray = None
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

        # Setting up parallel stuff
        rank, world_size, comm = mpi_params
        # Setting up numbers of things
        n_images = images.shape[0]

        if world_size == 1:
            n_nodes = path.shape[0]

        else:
            n_nodes = world_size + 2

        kappa = 1

        if initial_fe_prof is None:

            initial_fe_prof = 1.0 * np.random.randn(n_nodes)

        path_rank = prep_for_mpi(path, rank, world_size)

        prob_mat_rank = self.likelihood(path_rank, images, sigma)
        lenghts = np.array(comm.allgather(prob_mat_rank.size))

        prob_mat = np.empty((n_images * n_nodes))
        comm.Allgatherv(prob_mat_rank.T.flatten(), (prob_mat, lenghts))
        prob_mat = prob_mat.reshape(n_nodes, n_images).T

        if rank == 0:
            optimized_fe_prof = so.minimize(
                self.neg_log_posterior, initial_fe_prof, method="CG", args=(kappa, prob_mat)
            ).x

        else:
            optimized_fe_prof = np.empty_like(initial_fe_prof)

        comm.bcast(optimized_fe_prof, root=0)

        return optimized_fe_prof


class CryoVife(object):
    def __init__(self, images, sigma, beta):
        super().__init__()
        self.images = images
        self.sigma = sigma
        self.beta = beta

    def __call__(self, path, fe_prof):
        return CryoVIFE.grad_and_energy(path, fe_prof, self.images, self.sigma, self.beta)

    @staticmethod
    def grad_and_energy(
        path: np.ndarray, fe_prof: np.ndarray, images: np.ndarray, sigma: float, beta: float = 1
    ) -> Tuple[float, np.ndarray]:
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

        # raise Exception("CryoVIFE doesn't work at this moment, please use CryoBife")

        num_nodes = path.shape[0]

        # TODO: Think for a better name for weights
        weights = np.exp(-beta * fe_prof)  # density vec
        weights = weights / np.sum(weights)  # normalize, Eq.(8)

        path_image_dists = path - images[:, None]
        variance = sigma**2
        lognorm = -np.log(2 * np.pi * variance)
        log_prob_mat = -np.sum(path_image_dists**2, axis=-1) / (
            2 * variance
        )  # Unnormalized, we can add the normalization constant later.

        # Sum here since iid images
        variational_cost = -np.mean(np.dot(log_prob_mat, weights)) + num_nodes * lognorm
        entropy = np.dot(weights, np.log(weights))
        negative_elbo = variational_cost + entropy

        # Calculate gradient with respect to nodes
        grad = np.mean(path_image_dists, axis=0) * weights.reshape(-1, 1) / (variance)

        return negative_elbo, grad
