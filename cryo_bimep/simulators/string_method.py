"""Provide Langevin simulator for path optimization with cryo-bife"""
from typing import Callable, Tuple
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np

def arclength(path):

    x_diff = np.diff(path[:,0])
    y_diff = np.diff(path[:,1])
    arc_length = np.sum(np.sqrt((x_diff)**2+(y_diff)**2))

    return arc_length

def sum_arclength(path, norm_arc_length):

    frac_arc_length = 0
    x_diff = np.diff(path[:,0])
    y_diff = np.diff(path[:,1])

    Fracs_arc_length = []
    Updated_path_index = []

    Num_nodes = path.shape[0]

    for i in range(Num_nodes-1):

        frac_arc_length += np.sqrt((x_diff[i])**2+(y_diff[i])**2)

        if frac_arc_length >= norm_arc_length:

            Fracs_arc_length.append(frac_arc_length)
            Updated_path_index.append(i)
            frac_arc_length = 0
            
    return Updated_path_index, Fracs_arc_length

def run_string_method(initial_path):
    
    Cv = np.linspace(0,1,14)
    Cv2 = np.linspace(0,1,10000)
    Cspline1 = interpolate.CubicSpline(Cv,initial_path[:,0])
    Cspline2 = interpolate.CubicSpline(Cv,initial_path[:,1])
    
    Num_segments = initial_path.shape[0] - 1
    Path_spline = np.array([Cspline1(Cv2),Cspline2(Cv2)]).T
    norm_arc_length = arclength(Path_spline)/Num_segments
    New_nodes , _ = sum_arclength(Path_spline, norm_arc_length)
    
    reparam_curve = [[Cspline1(Cv2)[0],Cspline2(Cv2)[0]]] #First node fixed

    der_x_der_alpha = np.array(np.diff(Cspline1(Cv2))/np.diff(Cv2))
    der_y_der_alpha = np.array(np.diff(Cspline2(Cv2))/np.diff(Cv2))
    
    Tan_vec = np.array([der_x_der_alpha,der_y_der_alpha]).T
    Norm_Tan_vec = [[0,0]] #First node derivative

    for i in New_nodes:
    
        reparam_curve.append([Cspline1(Cv2)[i],Cspline2(Cv2)[i]])
        Norm_Tan_vec.append([Tan_vec[i,0], Tan_vec[i,1]])
    
    reparam_curve.append([Cspline1(Cv2)[-1],Cspline2(Cv2)[-1]]) #Last node fixed
    Norm_Tan_vec.append([0,0]) #Last node derivative
    reparam_curve = np.array(reparam_curve)
    Norm_Tan_vec = np.array(Norm_Tan_vec)
    Norm_Tan_vec = Norm_Tan_vec/np.sqrt(np.sum(Norm_Tan_vec**2))

    #print(Norm_Tan_vec.shape)
    
    #print(sum_arclength(Path_spline, arclength(Path_spline)/Num_segments), np.sum(_), len(_))
    #print(arclength(initial_path), arclength(initial_path)/Num_segments, Num_segments)
    #print(arclength(Path_spline), arclength(Path_spline)/Num_segments, Num_segments)
    #print(arclength(initial_path), arclength(reparam_curve))
    
    #plt.plot(initial_path[:,0],initial_path[:,1],'o' ,label = 'initial path')
    #plt.plot(Cspline1(Cv2),Cspline2(Cv2),'-', label = 'Spline String. path')
    #plt.plot(reparam_curve[:,0],reparam_curve[:,1],'*', label = 'Reparam. path')
    #plt.legend()
    #plt.show()    

    return reparam_curve, Norm_Tan_vec

#initial_path = np.loadtxt("../../example_data/Orange") - 1
#A, Tangent = run_string_method(initial_path)
#print(Tangent, np.sum(Tangent, axis=1), np.sum(np.sum(Tangent, axis=1)), Tangent.shape)
#plt.plot(Tangent[:,0],Tangent[:,1],'-*', label = 'Tangent vector')
#plt.legend()
#plt.show()