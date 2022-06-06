"""
Test that the evaluated gradient matches a numerical gradient.
Can be run using pytest, e.g. `pytest test_gradient.py`.
"""

import numpy as np
#import pytest
from pytest_easyMPI import mpi_parallel
from cryo_bimep.constraints import harm_rest_energy_and_grad
from cryo_bimep.utils import prep_for_mpi

@mpi_parallel(1)
def test_harm_rest_gradient_serial():

    from mpi4py import MPI
    "Test numerically harmonic restrain gradient"

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    mpi_params = (rank, world_size, comm)

    # Set up parameters for the test
    kappa = 1
    eps = 1e-7

    # Create random path, images, and free-energy profile
    n_nodes = 5
    test_path = np.random.randn(n_nodes, 2)
    test_ref_path = np.random.randn(n_nodes, 2)

    test_path_rank = prep_for_mpi(test_path, rank, world_size)

    ref_energy, _ = harm_rest_energy_and_grad(
        test_path_rank, test_ref_path, kappa, mpi_params
    )

    num_grad = np.zeros_like(test_path)
    an_grad = np.zeros_like(test_path)

    # First and last nodes are fixed
    for i in range(1, test_path.shape[0] - 1):
        for j in range(test_path.shape[1]):

            pert_path = test_path.copy()
            pert_path[i, j] += eps

            pert_path_rank = prep_for_mpi(pert_path, rank, world_size)

            comm.Barrier()
            pert_energy, pert_grad = harm_rest_energy_and_grad(
                pert_path_rank, test_ref_path, kappa, mpi_params
            )

            num_grad[i, j] = (pert_energy - ref_energy) / eps
            an_grad[i, j] = pert_grad[i, j]

    assert np.allclose(num_grad, an_grad)

def main():

    test_harm_rest_gradient_serial()

main()