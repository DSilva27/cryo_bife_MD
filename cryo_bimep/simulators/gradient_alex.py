"""Provide Langevin simulator for path optimization with cryo-bife"""
from typing import Callable, Tuple
from gradient import Grad
import numpy as np

def run_Gradient(
        initial_path: np.ndarray,
        grad_args: Tuple,
        steps: float,
        step_size: float) -> np.ndarray:
    """Run simulation using Euler-Maruyama algorithm.

    :param initial_path: Array with the initial values of the free-energy profile in each
        node of the path.
    :param fe_prof: Array with the values of the free-energy profile (FEP)
                    in each node of the path.
    :param grad_and_energy_func: Function that returns the energy and gradient.
                                 It must take as arguments (path, fe_profile, *args)
    :param grad_and_energy_args: extra arguments for grad_and_energy_func
    :param steps: Number of MALA steps to do
    :step_size: Step size for new proposals

    :returns: Last accepted path
    """

    # Calculate "old" variables
    sim_path = initial_path.copy()

    mask = np.ones_like(sim_path)

    mask[0] = np.zeros((2,))
    #mask[7] = np.zeros((2,))
    mask[-1] = np.zeros((2,))

    Grad_paths = np.zeros((steps+1, *initial_path.shape))
    Grad_paths[0] = initial_path

    for i in range(steps):

        # # Selecting which replica to update
        # path_index = [np.random.randint(0, initial_path.shape[0]), np.random.randint(0, 2)]
        # while path_index[0] in (0, 7, 13):
        #     path_index[0] = np.random.randint(0, initial_path.shape[0])

        # # Select which coordinate to update
        # path_index = tuple(path_index)


        # Calculate new proposal
        grad = Grad(sim_path, *grad_args)
        sim_path += step_size * grad * mask #(-step_size*grad + np.sqrt(2*step_size) * np.random.randn()) * mask

        #if _ %100 == 0:

        Grad_paths[i + 1] = sim_path
        #print(np.array(Grad_paths.shape))

    # returns last accepted path
    return Grad_paths


def main():

    gd_steps = 500 #100
    gd_step_size = 0.001 #0.00001
    gaussian_center = np.array([[6,6],[13,11],[7,15]]) - 1
    deepth_wells = np.array([2.27,1.93,1.55])
    T_inverse = 3
    w = 0.135 * T_inverse
    gd_args = (gaussian_center, deepth_wells, w)

    # Run path optimization
    
    initial_path = np.loadtxt("../../example_data/Orange") - 1
    #initial_path = np.loadtxt("example_data/O3") - 1
    
    path = run_Gradient(initial_path, gd_args,gd_steps,gd_step_size)
    np.save("../../example_data/Grad_path.npy", path)

    reference_path = np.loadtxt("../../example_data/Black")- 1
    #reference_path = np.loadtxt("example_data/Path_black-David") - 1
    return 0

if __name__ == "__main__":
    main()
