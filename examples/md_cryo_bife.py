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

brownian_traj = brownian_motion.run_brownian_sim(initial_path, steps=100, step_size=0.001)
brownian_path = brownian_traj[-1]

animate_simulation(
    brownian_traj,
    initial_path,
    images=None,
    ref_path=initial_path,
    anim_file=None,
)

path_after_string, _ = string_method.run_string_method(brownian_path)

plt.plot(path_after_string[:, 0], path_after_string[:, 1], ls=":", marker="*", label="after string")
plt.plot(brownian_path[:, 0], brownian_path[:, 1], ls="--", marker="o", label="before string")
plt.legend()
plt.show()


harm_params = (path_after_string, 1000)
restrained_traj = euler_maruyama.run_euler_maruyama(brownian_path, harm_rest_energy_and_grad, harm_params, steps=100)
restrained_path = restrained_traj[-1]

animate_simulation(
    restrained_traj,
    initial_path,
    images=None,
    ref_path=path_after_string,
    anim_file=None,
)

fe_prof = cryo_bife_obj.optimizer(restrained_path, images, cb_sigma)

cb_grad_energy_args = (0.5, 1, 1)

optimized_traj = stochastic_gd.run_stochastic_gd(restrained_path, fe_prof, CryoBife.grad_and_energy, cb_grad_energy_args, images, steps=1, step_size=0.0001, batch_size=int(images.shape[0] * 0.1))

optimized_path = optimized_traj[-1]

plt.plot(initial_path[:, 0], initial_path[:, 1], ls=":", marker="*", label="after string")
plt.plot(optimized_path[:, 0], optimized_path[:, 1], ls="--", marker="o", label="before string")
plt.legend()
plt.show()

#euler_maruyama.run_euler_maruyama(, harm_rest_energy_and_grad, )


#fe_prof = cryo_bife_obj.optimizer(new_path, images, cb_sigma)



