import numpy as np

from cryo_bimep import CryoBimep
from cryo_bimep.cryo_bife import CryoBife
from cryo_bimep.constraints import distances
from cryo_bimep.simulators import mala

def energy_and_grad_wrapper(path, fe_prof, cbife_args, dist_args):

    e_cbife, grad_cbife = CryoBife.grad_and_energy(path, fe_prof, *cbife_args)
    e_dist, grad_dist = distances.dist_energy_and_grad(path, *dist_args)

    e_total = e_cbife + e_dist
    grad_total = grad_cbife + grad_dist

    return e_total, grad_total

def main():

    np.random.seed(0)

    cryo_bimep_obj = CryoBimep()

    # Set simulator
    mala_steps = 100
    mala_step_size = 0.00001
    mala_args = (mala_steps, mala_step_size)
    cryo_bimep_obj.set_simulator(mala.run_mala, mala_args)

    # Set grad and energy func
    # cryo_bife args
    images = np.loadtxt("example_data/images.txt")
    cb_sigma = 0.5
    cb_beta = 1
    cb_kappa = 1
    cb_args = (images, cb_sigma, cb_beta, cb_kappa)

    # distance constraint args
    dc_kappa = 200*0
    dc_d0 = 2.5
    dc_args = (dc_kappa, dc_d0)

    # Energy and grad wrapper args
    energy_and_grad_args = (cb_args, dc_args)
    cryo_bimep_obj.set_grad_and_energy_func(energy_and_grad_wrapper, energy_and_grad_args)

    # Run path optimization
    initial_path = np.loadtxt("example_data/Orange")
    opt_steps = 10
    opt_fname = "paths.npy"
    cryo_bimep_obj.path_optimization(initial_path, images, opt_steps, opt_fname)

    return 0

if __name__ == "__main__":

    main()
