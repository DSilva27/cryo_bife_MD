"""
Test that the evaluated gradient matches a numerical gradient.
Can be run using pytest, e.g. `pytest test_gradient.py`.
"""

import numpy as np

# import pytest
from pytest_easyMPI import mpi_parallel
from cryo_bimep.cryo_bife import CryoBife
from cryo_bimep.utils import prep_for_mpi


@mpi_parallel(1)
def test_cryo_bife_grad():

    from mpi4py import MPI

    "Test numerically cryo-bife's gradient"

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    mpi_params = (rank, world_size, comm)

    # Set up parameters for the test
    eps = 1e-7
    cb_sigma = 0.5
    cb_beta = 1
    cb_kappa = 1

    # Create random path, images, and free-energy profile
    test_path = np.random.randn(world_size + 2, 2)
    test_images = np.random.randn(100, 2)
    test_fe = np.random.randn(world_size + 2)

    assert 4 < test_path.shape[0], "Too few nodes for test, use at least 5 (3 MPI ranks)"

    test_path_rank = prep_for_mpi(test_path, rank, world_size)

    ref_energy, _ = CryoBife.grad_and_energy(
        test_path_rank, test_fe, test_images, cb_sigma, cb_kappa, mpi_params, cb_beta
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
            pert_energy, pert_grad = CryoBife.grad_and_energy(
                pert_path_rank, test_fe, test_images, cb_sigma, cb_kappa, mpi_params, cb_beta
            )

            num_grad[i, j] = (pert_energy - ref_energy) / eps
            an_grad[i, j] = pert_grad[i, j]

    assert np.allclose(num_grad, an_grad)


def main():

    test_cryo_bife_grad()


main()
