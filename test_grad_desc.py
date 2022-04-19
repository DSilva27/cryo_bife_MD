import numpy as np

from cryo_bimep import CryoBimep
from cryo_bimep.cryo_bife import CryoBife
from cryo_bimep.constraints import distances
from cryo_bimep.simulators import stochastic_gd
from cryo_bimep.utils import animate_simulation

def energy_and_grad_wrapper(path, fe_prof, images, cbife_args, dist_args):

    e_cbife, grad_cbife = CryoBife.grad_and_energy(path, fe_prof, images, *cbife_args)
    e_dist, grad_dist = distances.dist_energy_and_grad(path, *dist_args)

    e_total = e_cbife + e_dist
    grad_total = grad_cbife + grad_dist

    return e_total, grad_total

def main():

    cryo_bimep_obj = CryoBimep()

    # Set simulator
    images = np.loadtxt("test_data/images_3well.txt")
    gd_steps = 10
    gd_step_size = 0.00001
    gd_batch_size = int(images.shape[0] * 0.1)
    gd_args = (images, gd_steps, gd_step_size, gd_batch_size)
    cryo_bimep_obj.set_simulator(stochastic_gd.run_stochastic_gd, gd_args)

    # Set grad and energy func
    # cryo_bife args
    cb_sigma = 0.5
    cb_beta = 1
    cb_kappa = 1
    cb_args = (cb_sigma, cb_beta, cb_kappa)

    # distance constraint args
    dc_kappa = 1000
    dc_d0 = 0.0
    dc_args = (dc_kappa, dc_d0)

    # Energy and grad wrapper args
    energy_and_grad_args = (cb_args, dc_args)
    cryo_bimep_obj.set_grad_and_energy_func(energy_and_grad_wrapper, energy_and_grad_args)

    # Run path optimization
    initial_path = np.loadtxt("test_data/mid_node_far") - 1
    #initial_path[:, [0,1]] = initial_path[:, [1,0]]
    opt_steps = 5
    opt_fname = "paths.npy"

    print("Starting path optimization using sthochastic gradient descent")
    print(f"Optimization iteratons: {opt_steps}, gd steps: {gd_steps}")
    traj = cryo_bimep_obj.path_optimization(initial_path, images, opt_steps, opt_fname)
    print("Optimization finished")
    print("*"*80)
    print("Animating trajectory")
    animate_simulation(traj, initial_path, images)


    return 0

if __name__ == "__main__":
    main()
