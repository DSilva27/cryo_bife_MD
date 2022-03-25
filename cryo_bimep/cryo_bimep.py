"Provides implementation of cryo-BIMEP"
from typing import Callable, Tuple
import numpy as np
from tqdm import tqdm

from cryo_bimep.cryo_bife import CryoBife

class CryoBimep(CryoBife):
    """CryoBimep provides the methodology to optimize a path using
       a simulator and cryo-bife iteratively."""


    def __init__(self):
        """Constructor. Initializes cryo-bife object.
        """
        CryoBife.__init__(self)

        self._simulator = None
        self._sim_args = None

        self._grad_and_energy_func = None
        self._grad_and_energy_args = None

    def set_simulator(
            self,
            sim_func: Callable,
            sim_args: Tuple):
        """Defines the simulator to be used.

        :param sim_func: 
        """

        self._simulator = sim_func
        self._sim_args = sim_args
    
    def set_grad_and_energy_func(self, grad_and_energy_func, args):

        self._grad_and_energy_func = grad_and_energy_func
        self._grad_and_energy_args = args

    def path_optimization(self, initial_path, images, steps, paths_fname = None):

        sigma = 0.5
        tol = 1e-2
        paths = np.zeros((steps+1, *initial_path.shape))
        paths[0] = initial_path

        curr_path = initial_path.copy()

        for i in tqdm(range(steps)):

            fe_prof = self.optimizer(curr_path, images, sigma)
            curr_path = self._simulator(curr_path, fe_prof, self._grad_and_energy_func, 
                                        self._grad_and_energy_args, *self._sim_args)

            paths[i+1] = curr_path

            if np.sum(abs(paths[i+1] - paths[i]))/initial_path.size < tol:
                paths = paths[:i+1]
                break

        if paths_fname is not None:

            if ".txt" in paths_fname:
                np.savetxt(f"{paths_fname}", paths)

            elif ".npy" in paths_fname:
                np.save(f"{paths_fname}", paths)

            else:
                print(f"Unknown file extension, saving as npy instead")
                np.save(f"{paths_fname.partition('.')[0]}.npy", paths)

        return 0



