"""
Test that the evaluated gradient matches a numerical gradient.
Can be run using pytest, e.g. `pytest test_gradient.py`.
"""

from unicodedata import numeric
from .cryo_bife import CryoVIFE
import numpy as np

def test_CryoVIFE_grad():
    np.random.seed(8675309)

    # Set up parameters for the run
    eps = 1e-7
    sigma = 0.1
    beta = 1.
    images = np.random.randn(100, 2)
    path = np.random.randn(10, 2)
    fe = -np.log(np.random.rand(10))


    e_ref, g_ref = CryoVIFE.grad_and_energy(
        path, fe, images, sigma, beta)

    # Evaluate numerical gradient with fwd difference formula
    for i in range(10):
        for dim in range(2):
            perturbed_path = np.copy(path)
            perturbed_path[i, dim] += eps

            e_delta = CryoVIFE.grad_and_energy(
                perturbed_path, fe, images, sigma, beta)[0]

            numerical_grad = (e_delta - e_ref)/ eps

            assert(np.linalg.norm(numerical_grad - g_ref[i, dim]) < 1e-5)
