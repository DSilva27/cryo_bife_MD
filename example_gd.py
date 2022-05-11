import numpy as np

from cryo_bimep import CryoBimep
from cryo_bimep import cryo_bife
from cryo_bimep.cryo_bife import CryoBife
from cryo_bimep.constraints import distances
from cryo_bimep.simulators import stochastic_gd, string_stochastic

def energy_and_grad_wrapper(path, fe_prof, images, cbife_args, dist_args):

    e_cbife, grad_cbife = CryoBife.grad_and_energy(path, fe_prof, images, *cbife_args)
    e_dist, grad_dist = distances.dist_energy_and_grad(path, *dist_args)

    e_total = e_cbife + 0*e_dist
    grad_total = grad_cbife + grad_dist

    return e_total, grad_total

def main():

    cryo_bimep_obj = CryoBimep()

    # Set simulator
    # images = np.loadtxt("example_data/2-wells_images.txt")
    images = np.loadtxt("example_data/images.txt")
    gd_steps = 10 #100
    gd_step_size = 0.0001
    gd_batch_size = int(images.shape[0] * 0.1)
    gd_args = (images, gd_steps, gd_step_size, gd_batch_size)
    #cryo_bimep_obj.set_simulator(stochastic_gd.run_stochastic_gd, gd_args)
    cryo_bimep_obj.set_simulator(string_stochastic.run_string_stochastic, gd_args)

    # Set grad and energy func
    # cryo_bife args
    cb_sigma = 0.5
    cb_beta = 1
    cb_kappa = 1
    cb_args = (cb_sigma, cb_beta, cb_kappa)

    # distance constraint args
    dc_kappa = 200
    dc_d0 = 0#0.5
    dc_args = (dc_kappa, dc_d0)

    # Energy and grad wrapper args
    energy_and_grad_args = (cb_args, dc_args)
    cryo_bimep_obj.set_grad_and_energy_func(energy_and_grad_wrapper, energy_and_grad_args)

    # Run path optimization
    #initial_path = np.loadtxt("example_data/Path_orange-David") - 1
    #initial_path[:, [0,1]] = initial_path[:, [1,0]]
    
    initial_path = np.loadtxt("example_data/Orange") - 1
    #initial_path = np.loadtxt("example_data/O3") - 1
    opt_steps = 300 #1500
    opt_fname = "string_paths_sgd_toy_system.npy"

    #String_path, _dummy = string_method.run_string_method(initial_path)
    cryo_bimep_obj.path_optimization(initial_path, images, opt_steps, opt_fname)
    #cryo_bimep_obj.path_optimization(String_path,images, opt_steps, opt_fname)

    reference_path = np.loadtxt("example_data/Black") - 1
    #reference_path = np.loadtxt("example_data/Path_black-David") - 1
    CB = CryoBife()
    black_fe_prof, black_log_posterior = CryoBife.optimizer(CB,reference_path, images, cb_sigma)
    np.savetxt('3-wells_Reference_FE', black_fe_prof)
    np.savetxt('3-wells_Reference_LogPost', np.array([black_log_posterior]))

    return 0

if __name__ == "__main__":
    main()
