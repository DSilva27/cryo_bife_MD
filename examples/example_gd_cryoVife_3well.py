import numpy as np

from cryo_bimep import CryoBimep
from cryo_bimep.cryo_bife import CryoVife
from cryo_bimep.constraints import distances
from cryo_bimep.simulators import stochastic_gd
from cryo_bimep.utils import animate_simulation


def energy_and_grad_wrapper(path, fe_prof, images, cbife_args, dist_args):

    e_cvife, grad_cvife = CryoVife.grad_and_energy(path, fe_prof, images, *cbife_args)
    e_dist, grad_dist = distances.dist_energy_and_grad(path, *dist_args)

    e_total = e_cvife + e_dist
    grad_total = grad_cvife + grad_dist

    return e_total, grad_total


def main():

    cryo_bimep_obj = CryoBimep()

    # Set simulator
    images = np.loadtxt("3_well_data/images.txt")
    gd_steps = 100
    gd_step_size = 0.001
    gd_batch_size = None  # int(images.shape[0] * 0.1)
    gd_args = (images, gd_steps, gd_step_size, gd_batch_size)
    cryo_bimep_obj.set_simulator(stochastic_gd.run_stochastic_gd, gd_args)

    # Set grad and energy func
    # cryo_bife args
    # cb_sigma = 0.5
    # cb_beta = 1
    # cb_kappa = 1
    # cb_args = (cb_sigma, cb_beta, cb_kappa)

    # cryo_vife args
    cv_sigma = 0.5
    cv_beta = 1
    cv_args = (cv_sigma, cv_beta)

    # distance constraint args
    dc_kappa = 1
    dc_d0 = 0.0
    dc_args = (dc_kappa, dc_d0)

    # Energy and grad wrapper args
    energy_and_grad_args = (cv_args, dc_args)
    cryo_bimep_obj.set_grad_and_energy_func(energy_and_grad_wrapper, energy_and_grad_args)

    # Run path optimization
    # This path has the node in the middle far away from where it's supposed to be
    initial_path = np.loadtxt("3_well_data/initial_path_far_mid_node") - 1
    opt_steps = 30
    opt_fname = None  # "paths.npy"

    print("Starting path optimization using sthochastic gradient descent")
    print(f"Optimization iteratons: {opt_steps}, gd steps: {gd_steps}")
    traj = cryo_bimep_obj.path_optimization(initial_path, images, opt_steps, opt_fname)
    print("Optimization finished")
    print("*" * 80)
    print("Animating trajectory")
    animate_simulation(traj, initial_path, images=None)

    return 0


if __name__ == "__main__":
    main()
