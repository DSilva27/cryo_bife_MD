import time
import numpy as np
import scipy.optimize as so
import matplotlib.pyplot as plt
from numba import config,jit ,njit, threading_layer

"""
Qué variables necesita definir el programa con antelación?:

sigma = 0.5 ---> ruido del toy system / varianza de las imágenes gaussianas
kappa = 1 ---> para p(G)
kappa_2 = 90 ---> para la penalización en la distancia
alpha = 60 ---> para la penalización sobre la derivada
beta = 1.0 ---> para mejorar el "sampling" sobre la superficie 2D
h = 0.001 ---> factor de Langevin
MALA_old = -999999999999 ---> para arrancar el programa

G_acc = [] ---> Energias libres aceptadas
LogPost_acc = [] ---> LogPosterior aceptadas
Path_acc = [] ---> caminos aceptados

LMC_steps = 10 ---> número de pasos del LMC
num_rejected = 0 ---> número de caminos rechazados
"""

##NUMBA_THREADING_LAYER='omp'
##config.THREADING_LAYER = 'threadsafe'
##@njit(parallel=True)
##@jit
def integrated_prior(G):
    
    mathcal_G = sum(np.diff(G)**2)
    log_prior = np.log(1/mathcal_G**2)    # note kappa scales *log* prior
    
    return log_prior

##NUMBA_THREADING_LAYER='omp'
##config.THREADING_LAYER = 'threadsafe'
##@njit(parallel=True)
##@jit
def post_prob(path, gauss_image_vector, sigma):

    number_of_nodes = path.shape[0]
    number_of_images = gauss_image_vector.shape[0]
    prob_matrix = np.zeros((number_of_images, number_of_nodes))

    norm = (1 / (2 * np.pi * sigma**2))
    
    prob_matrix = norm * np.exp(-((path[:,0][:,None] - gauss_image_vector[:,0])**2 \
                                    + (path[:,1][:,None] - gauss_image_vector[:,1])**2) \
                                      /  (2 * sigma**2))

    return prob_matrix.T

##NUMBA_THREADING_LAYER='omp'
##config.THREADING_LAYER = 'threadsafe'
##@njit(parallel=True)
##@jit
def neglogpost_cryobife(G, kappa, Pmat, log_prior_fxn=None):

    if log_prior_fxn is None: # Default is the prior from the paper

        log_prior_fxn = integrated_prior

    log_prior = kappa * log_prior_fxn(G)
    
    rho = np.exp(-G) #density vec
    rho = rho / np.sum(rho) #normalize, Eq.(8)

    log_likelihood = np.sum(np.log(np.dot(Pmat, rho))) # sum here since iid images; logsumexp
    neg_log_posterior = -(log_likelihood + log_prior)

    return neg_log_posterior

##NUMBA_THREADING_LAYER='omp'
##config.THREADING_LAYER = 'threadsafe'
##@njit(parallel=True)
##@jit
def gradient_cryo_bife(path, G, images, prob_mat, sigma, beta=1):

    grad = np.zeros(path.shape)

    rho = np.exp(-beta * G)
    rho = rho / np.sum(rho)

    Z = np.sum(prob_mat * rho, axis=1)
    weighted_prob_mat = rho[:,None] * prob_mat.T / Z

    grad[:,0] = np.sum((images[:,0] - path[:,0][:,None]) * weighted_prob_mat, axis=1)
    grad[:,1] = np.sum((images[:,1] - path[:,1][:,None]) * weighted_prob_mat, axis=1)


    return -1 / sigma**2 * grad

#NUMBA_THREADING_LAYER='omp'
##config.THREADING_LAYER = 'threadsafe'
##@njit(parallel=True)
##@jit
def models_dist(new_path):  #Penalization respect to the distance of each node respect to their nearest neighbor.

    dist = 0

    for i in range(1,new_path.shape[0]-1):

        dist += (np.sqrt((new_path[i-1][0] - new_path[i][0])**2 + (new_path[i-1][1] - new_path[i][1])**2) \
              - np.sqrt((new_path[i+1][0] - new_path[i][0])**2 + (new_path[i+1][1] - new_path[i][1])**2)) #**2

    return(dist)

##NUMBA_THREADING_LAYER='omp'
##config.THREADING_LAYER = 'threadsafe'
##@njit(parallel=True)
##@jit
def models_der(new_path):  #Derivative condition checking the softness of the path.

    der = 0
    for i in range(1,new_path.shape[0]-1):

        th1=np.arctan2(float(new_path[i][1]-new_path[i-1][1]),float(new_path[i][0]-new_path[i-1][0]))
        th2=np.arctan2(float(new_path[i+1][1]-new_path[i][1]),float(new_path[i+1][0]-new_path[i][0]))

        der += (th1-th2)**2

    return(der)

##NUMBA_THREADING_LAYER='omp'
##config.THREADING_LAYER = 'threadsafe'
##@njit(parallel=True)
##@jit
def langevin(path,images,LMC_steps):

    kappa = 1
    sigma = 0.5

    kappa_2 = 90
    alpha = 60
    beta = 1.0
    h = 0.001

    #LMC_steps = 10
    num_rejected = 0

    G_acc = []
    MALA_acc = []
    LogPost_acc = []
    Path_x_coords_acc = []
    Path_y_coords_acc = []

    Old_path = np.copy(path)
    model = Old_path.shape[0]
    index = Old_path.shape[1]

    MALA_num = -9999999999999

    for step in range(LMC_steps):

        print(f'LMC step {step}')
        New_path = np.copy(Old_path)
        value = False
        while value == False:

            model_index = np.random.randint(0,model)
            if(model_index==0 or model_index==7 or model_index==13):
                print(f'Node {model_index} is fixed...choosing different node')
                continue

            else: value = True

        coord_index = np.random.randint(0, 2)
        print(f'model = {model_index} ; coord = {coord_index}')

        Xi = np.random.randn() #(model, index)
        G_rand = 1.0 * np.random.randn(model)
        prob_mat = post_prob(Old_path, images, sigma)
        G_op = so.minimize(neglogpost_cryobife, G_rand ,method='CG',args=(kappa,prob_mat,))
        log_post_old = -1*G_op.fun
        free_energy_old = G_op.x

        gradient_old = gradient_cryo_bife(Old_path, free_energy_old, images, prob_mat, sigma)
        New_path[model_index,coord_index] += -h*gradient_old[model_index,coord_index] + np.sqrt(2*h)*Xi #[model_index,coord_index]

        der = models_der(New_path)
        dist = models_dist(New_path)
  
        MALA_den = -log_post_old - np.sum( (New_path - Old_path + h * gradient_old)**2 ) - kappa_2*dist - alpha*der

        aa=0
        rr = np.log(np.random.random())

        if (MALA_den > MALA_num):

            aa += 1
            Old_path = np.copy(New_path)
            MALA_num = MALA_den
            MALA_acc.append(MALA_den)

            prob_mat_new = post_prob(New_path, images, sigma)
            G_op_new = so.minimize(neglogpost_cryobife, G_rand ,method='CG',args=(kappa,prob_mat_new,))
            log_post_new = -1*G_op_new.fun
            free_energy_new = G_op_new.x

            Path_x_coords_acc.append(New_path[:,0])
            Path_y_coords_acc.append(New_path[:,1])
            G_acc.append(free_energy_new)
            LogPost_acc.append(log_post_new)

            print('Path Accepted ')

        elif (rr < -(MALA_num-MALA_den)*beta):

            aa += 2
            Old_path = np.copy(New_path)
            MALA_num = MALA_den
            MALA_acc.append(MALA_den)

            prob_mat_new = post_prob(New_path, images, sigma)
            G_op_new = so.minimize(neglogpost_cryobife, G_rand ,method='CG',args=(kappa,prob_mat_new,))
            log_post_new = -1*G_op_new.fun
            free_energy_new = G_op_new.x

            Path_x_coords_acc.append(New_path[:,0])
            Path_y_coords_acc.append(New_path[:,1])
            G_acc.append(free_energy_new)
            LogPost_acc.append(log_post_new)

            print('Path Accepted ')

        else:

            Old_path = Old_path
            MALA_den = MALA_den
            num_rejected += 1
            print('Path Rejected')

        print('ProbsPilar', aa, MALA_den,MALA_num, -dist, -der,log_post_new)

        #print(f'Valor Neg_logpost = {-log_post}')
        #print(f'Valor MALA_New = {MALA_New}') 

        """
        plt.figure()
        plt.plot([i for i in range(len(free_energy))],free_energy,'-o', label = 'free energy')
        plt.legend()
        plt.show()
        """
    G_acc = np.array(G_acc)
    MALA_acc = np.array(MALA_acc)
    LogPost_acc = np.array(LogPost_acc)
    Path_x_coords_acc = np.array(Path_x_coords_acc)
    Path_y_coords_acc = np.array(Path_y_coords_acc)

    np.savetxt('G_acc',G_acc)
    np.savetxt('MALA_acc',MALA_acc)
    np.savetxt('LogPost_acc',LogPost_acc)
    np.savetxt('Path_x_coords_acc',Path_x_coords_acc)
    np.savetxt('Path_y_coords_acc',Path_y_coords_acc)
    
    print('Total rejected paths =',num_rejected)

    return(New_path)

##NUMBA_THREADING_LAYER='omp'
##config.THREADING_LAYER = 'threadsafe'
##@njit(parallel=True)
##@jit
def main():

    start_time = time.time()

    LMC_steps = 20
    path = np.loadtxt("data/Orange")
    images = np.loadtxt("data/images.txt")
    images = images - 10*np.ones(images.shape)
    path_test = np.copy(path) - 11*np.ones(path.shape)

    langevin(path_test,images,LMC_steps)

    print("The program takes %s seconds for %s LMC steps---" % ((time.time() - start_time), LMC_steps))
    
    return 0

if __name__ == "__main__":
    
    main()
