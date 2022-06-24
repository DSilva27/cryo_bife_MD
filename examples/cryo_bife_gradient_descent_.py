import numpy as np
import matplotlib.pyplot as plt

from cryo_bimep.simulators import brownian_motion, euler_maruyama, stochastic_gd
from cryo_bimep.string_method import string_method
from cryo_bimep.cryo_bife import CryoBife
from cryo_bimep.utils import animate_simulation

from cryo_bimep.constraints.harm_restrain import harm_rest_energy_and_grad

"""
1. Unbiased MD (swarms = 1)
2. Reparametrize with string method -> new path (or new nodes)
3. Restrained MD using the new path (move atomic structures to the new nodes)
4. Optimize the path using cryo-bife (using the cryo-em images)  (gradient descent or something)
5. Reparametrize again
6. restrained again

Repeat

1 -> 4 -> 5 -> 6 (repeat 4-6 if necessary)

H(x, y)
"""


cryo_bife_obj = CryoBife()

initial_path = np.loadtxt("3_well_data/initial_path_far_mid_node") - 1
images = np.loadtxt("3_well_data/images.txt")
cb_sigma = 0.5
cb_kappa = 1
cb_beta = 1
cb_grad_energy_args = (cb_sigma, cb_kappa, cb_beta)
fe_prof = None

opt_steps = 1
gd_steps = 10
gd_step_size = 0.0001
gd_batch_size = int(images.shape[0] * 0.1)

full_traj = np.zeros((opt_steps * gd_steps, *initial_path.shape))
current_path = np.copy(initial_path)
all_fe_prof = np.zeros((opt_steps, initial_path.shape[0]))

for j in range(opt_steps):

    fe_prof = cryo_bife_obj.optimizer(current_path, images, cb_sigma, fe_prof)
    optimized_traj = stochastic_gd.run_stochastic_gd(
        current_path,
        fe_prof,
        CryoBife.grad_and_energy,
        cb_grad_energy_args,
        images,
        steps=gd_steps,
        step_size=gd_step_size,
        batch_size=gd_batch_size,
    )
    current_path = optimized_traj[-1]
    full_traj[j * gd_steps : (j + 1) * gd_steps] = optimized_traj
    all_fe_prof[j] = fe_prof
np.save("Full_traj.npy", full_traj)
np.save("All_fe_prof.npy", all_fe_prof)

animate_simulation(
    full_traj,
    initial_path,
    images=images,
    ref_path=initial_path,
    anim_file=None,
)
