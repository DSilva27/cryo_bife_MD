import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from cryo_bimep.simulators.euler_maruyama import run_euler_maruyama

from cryo_bimep.cryo_bife import optimize_free_energy
from cryo_bimep.utils import animate_simulation

from cryo_bimep.constraints.harm_restrain import harm_rest_energy_and_grad
from cryo_bimep.cryo_cv import calc_energy_and_grad as cryo_cv_energy_and_grad

"""
This example does Erik's idea using the following workflow: 

1. Do cryo-bife to get the free energy
2. DO Restrained MD using the new path (sample from \pi) 
3. Do gradient descent  to optimize the path using Erik's idea
4. Repeat
"""

initial_path = np.loadtxt("3_well_data/initial_path_far_mid_node") - 1
images = np.loadtxt("3_well_data/images.txt")

# Set parameters for restrained simulation
SPRING_CONSTANT = 10.0
REST_SIM_STEPS = 1000
REST_SIM_STEP_SIZE = 0.001
REST_SIM_STRIDE = 10  # If stride is None (default), only output the last path. If you want all paths set to 1

# Set parameters for gradient descent
GD_STEP_SIZE = 0.00001

# Set parameters for optimization workflow
OPT_STEPS = 100
CB_SIGMA = 1
opt_traj = np.zeros((OPT_STEPS, *initial_path.shape))

fe_prof = None
current_path = initial_path.copy()
harm_params = (current_path, SPRING_CONSTANT)
fe_prof = optimize_free_energy(current_path, images, CB_SIGMA, fe_prof)

for i in tqdm(range(OPT_STEPS)):

    restrained_traj = run_euler_maruyama(
        current_path,
        harm_rest_energy_and_grad,
        harm_params,
        steps=REST_SIM_STEPS,
        step_size=REST_SIM_STEP_SIZE,
        stride=REST_SIM_STRIDE,
    )

    restrained_traj = np.transpose(restrained_traj, axes=[1, 0, 2])

    _, gradient = cryo_cv_energy_and_grad(
        current_path, restrained_traj, fe_prof, images, SPRING_CONSTANT
    )

    current_path -= GD_STEP_SIZE * gradient
    opt_traj[i] = current_path

    harm_params = (current_path, SPRING_CONSTANT)


animate_simulation(
    opt_traj,
    initial_path,
    images=None,
    ref_path=None,
    anim_file=None,
)
