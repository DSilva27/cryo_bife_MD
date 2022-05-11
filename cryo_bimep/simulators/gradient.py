"""Provide functions related to Cryo-BIFE"""
from typing import Callable, Tuple
import numpy as np


def Grad(path: np.ndarray,
         gaussian_center: np.ndarray,
         deepth_wells: np.ndarray,
         w: float) -> np.ndarray:
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
     # Calculate gradient
    grad = np.zeros_like(path)
    norm = deepth_wells[0] * np.exp(-( ( (path[:,0] - gaussian_center[0,0])**2 + (path[:,1] - gaussian_center[0,1])**2 ) / 2*w**2 )) \
    + deepth_wells[1] * np.exp(-( ( (path[:,0] - gaussian_center[1,0])**2 + (path[:,1] - gaussian_center[1,1])**2 ) / 2*w**2 )) \
    + deepth_wells[2] * np.exp(-( ( (path[:,0] - gaussian_center[2,0])**2 + (path[:,1] - gaussian_center[2,1])**2 ) / 2*w**2 ))


    grad[:,0] = (-deepth_wells[0]*(path[:,0] - gaussian_center[0,0])/w**2) * np.exp(-( ( (path[:,0] - gaussian_center[0,0])**2 + (path[:,1] - gaussian_center[0,1])**2 ) / 2*w**2 )) \
    + (-deepth_wells[1]*(path[:,0] - gaussian_center[1,0])/w**2) * np.exp(-( ( (path[:,0] - gaussian_center[1,0])**2 + (path[:,1] - gaussian_center[1,1])**2 ) / 2*w**2 )) \
    + (-deepth_wells[2]*(path[:,0] - gaussian_center[2,0])/w**2) * np.exp(-( ( (path[:,0] - gaussian_center[2,0])**2 + (path[:,1] - gaussian_center[2,1])**2 ) / 2*w**2 ))


    grad[:,1] = (-deepth_wells[0]*(path[:,1] - gaussian_center[0,1])/w**2) * np.exp(-( ( (path[:,0] - gaussian_center[0,0])**2 + (path[:,1] - gaussian_center[0,1])**2 ) / 2*w**2 )) \
     + (-deepth_wells[1]*(path[:,1] - gaussian_center[1,1])/w**2) * np.exp(-( ( (path[:,0] - gaussian_center[1,0])**2 + (path[:,1] - gaussian_center[1,1])**2 ) / 2*w**2 )) \
     + (-deepth_wells[2]*(path[:,1] - gaussian_center[2,1])/w**2) * np.exp(-( ( (path[:,0] - gaussian_center[2,0])**2 + (path[:,1] - gaussian_center[2,1])**2 ) / 2*w**2 )) 

    for i in range(14):
         grad[i,:] = grad[i,:]/norm[i]

    return grad
