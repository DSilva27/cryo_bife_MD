"Provides constraint on distances between nodes of the path"
import numpy as np

def dist_energy_and_grad(path, kappa_2, d0):

    energy_dist = np.sum((np.sqrt(np.sum((path[:-1] - path[1:])**2, axis=1)) - d0)**2)
    energy_dist *= 0.5 * kappa_2

    def distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2, axis=-1))

    grad_dist = np.zeros_like(path)

    grad_dist[0] = (1 - d0 / distance(path[0], path[1])) *\
                   (path[0] - path[1])
    grad_dist[-1] = (1 - d0 / distance(path[-1], path[-2])) *\
                    (path[-1] - path[-2])

    grad_dist[1:-1] = (1 - d0 / distance(path[1:-1], path[2:])[:,None]) *\
                      (path[1:-1] - path[2:]) +\
                      (1 - d0 / distance(path[1:-1], path[:-2])[:,None]) *\
                      (path[1:-1] - path[:-2])

    grad_dist *= kappa_2

    return energy_dist, grad_dist
