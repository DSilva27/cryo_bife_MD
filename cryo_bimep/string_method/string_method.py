"""Provide implementation of the string method"""
from scipy import interpolate
import numpy as np


def arclength(path):

    path_diff = np.diff(path, axis=0)
    arc_length = np.sum(np.sqrt((path_diff**2).sum(axis=1)))

    return arc_length


def find_new_nodes(path_string, n_nodes, norm_arc_length):

    path_diff = np.diff(path_string, axis=0)
    cumsum_arc_length = np.cumsum(np.sqrt((path_diff**2).sum(axis=1)))

    new_nodes = np.zeros((n_nodes,), dtype=int)
    new_nodes[-1] = -1

    for i in range(1, n_nodes - 1):

        index = np.argmax(cumsum_arc_length >= norm_arc_length)
        new_nodes[i] = index + new_nodes[i - 1]

        cumsum_arc_length = cumsum_arc_length[index:] - cumsum_arc_length[index]

    return new_nodes


def run_string_method(path):
    """
    Reparametrice a path using the string method

    Parameters
    ----------
    path: path to be reparametriced

    Returns
    ---------
    new_path: reparametriced path
    """

    n_segments = path.shape[0] - 1

    x_calc_spline = np.linspace(0, 1, path.shape[0])
    cspline1 = interpolate.CubicSpline(x_calc_spline, path[:, 0])
    cspline2 = interpolate.CubicSpline(x_calc_spline, path[:, 1])

    x_eval_spline = np.linspace(0, 1, 10000)
    path_string = np.array([cspline1(x_eval_spline), cspline2(x_eval_spline)]).T

    norm_arc_length = arclength(path_string) / n_segments
    new_nodes = find_new_nodes(path_string, path.shape[0], norm_arc_length)

    new_path = path_string[new_nodes]

    path_string_der = np.array([cspline1(x_eval_spline, 1), cspline2(x_eval_spline, 1)]).T

    tangent_to_new_path = np.zeros_like(new_path)
    tangent_to_new_path[1:-1] = path_string_der[new_nodes[1:-1]]
    tangent_to_new_path /= np.linalg.norm(tangent_to_new_path)

    return new_path, tangent_to_new_path
