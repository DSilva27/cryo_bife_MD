import numpy as np

from cryo_bimep import CryoBimep
from cryo_bimep.cryo_bife import CryoBife
from cryo_bimep.constraints import distances
from cryo_bimep.simulators import stochastic_gd_string
from cryo_bimep.utils import animate_simulation

from mpi4py import MPI


def energy_and_grad_wrapper(path, fe_prof, images, cbife_args, dist_args):

    e_cvife, grad_cvife = CryoBife.grad_and_energy(path, fe_prof, images, *cbife_args)
    e_dist, grad_dist = distances.dist_energy_and_grad(path, *dist_args)

    e_total = e_cvife + e_dist
    grad_total = grad_cvife + grad_dist

    return e_total, grad_total


def main():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    mpi_params = (rank, world_size, comm)

    cryo_bimep_obj = CryoBimep()

    # Set simulator
    images = np.loadtxt("3_well_data/images.txt")
    gd_steps = 100
    gd_step_size = 0.00001
    gd_batch_size = int(images.shape[0] * 0.1)
    gd_args = (mpi_params, images, gd_steps, gd_step_size, gd_batch_size)
    cryo_bimep_obj.set_simulator(stochastic_gd_string.run_stochastic_gd_string, gd_args)

    # Set grad and energy func
    # cryo_bife args
    cb_sigma = 0.5
    cb_beta = 1
    cb_kappa = 1
    cb_args = (cb_sigma, cb_kappa, mpi_params, cb_beta)

    # distance constraint args
    dc_kappa = 100
    dc_d0 = 0.0
    dc_args = (dc_kappa, dc_d0, mpi_params)

    # Energy and grad wrapper args
    energy_and_grad_args = (cb_args, dc_args)
    cryo_bimep_obj.set_grad_and_energy_func(energy_and_grad_wrapper, energy_and_grad_args)

    # Run path optimization
    # This path has the node in the middle far away from where it's supposed to be
    #initial_path = np.loadtxt("3_well_data/initial_path_far_mid_node") - 1
    initial_path = np.load("tight_path.npy")

    assert (world_size + 2 == initial_path.shape[0]) or (world_size == 1), "Wrong world size"

    opt_steps = 10
    opt_fname = f"traj_bife_and_harm_{dc_kappa}.npy"

    if rank == 0:
        print("Starting path optimization using sthochastic gradient descent")
        print(f"Optimization iteratons: {opt_steps}, gd steps: {gd_steps}")

    traj = cryo_bimep_obj.path_optimization(initial_path, images, opt_steps, mpi_params, opt_fname)
    # traj = np.load(opt_fname)

    if rank == 0:
        print("Optimization finished")
        print("*" * 80)

    if rank == 0:
        print("Animating trajectory")
        # animate_simulation(traj, initial_path, images=images, ref_path=np.loadtxt("3_well_data/ref_path") - 1, anim_file="3well_bife_no_const")
        animate_simulation(
            traj,
            initial_path,
            images=None,
            ref_path=np.loadtxt("3_well_data/ref_path") - 1,
            anim_file=f"traj_bife_and_harm_{dc_kappa}",
        )

    return 0


if __name__ == "__main__":
    main()
