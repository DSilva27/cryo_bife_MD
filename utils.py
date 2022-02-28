import numpy as np
import time
import numpy as np
import scipy.optimize as so
import matplotlib.pyplot as plt

def integrated_prior(G):
    
    mathcal_G = sum(np.diff(G)**2)
    new_log_post = np.log(1/mathcal_G**2)    # note kappa scales *log* prior
    
    return new_log_post

def post_prob(path, gauss_image_vector, sigma):

    number_of_nodes = path.shape[0]
    number_of_images = gauss_image_vector.shape[0]
    prob_matrix = np.zeros((number_of_images, number_of_nodes))

    norm = (1 / (2 * np.pi * sigma**2))
    
    prob_matrix = norm * np.exp(-((path[:,0][:,None] - gauss_image_vector[:,0])**2 \
                                    + (path[:,1][:,None] - gauss_image_vector[:,1])**2) \
                                      /  (2 * sigma**2))

    return prob_matrix.T

def neglogpost_cryobife(G, kappa, Pmat, new_log_post_fxn=None):

    if new_log_post_fxn is None: # Default is the prior from the paper

        new_log_post_fxn = integrated_prior

    new_log_post = kappa * new_log_post_fxn(G)
    
    rho = np.exp(-G) #density vec
    rho = rho / np.sum(rho) #normalize, Eq.(8)

    log_likelihood = np.sum(np.log(np.dot(Pmat, rho))) # sum here since iid images; logsumexp
    neg_log_posterior = -(log_likelihood + new_log_post)

    return neg_log_posterior # check new_log_post sign error?

def gradient_cryo_bife(path, G, images, prob_mat, sigma, beta=1):

    grad = np.zeros(path.shape)

    rho = np.exp(-beta * G)
    rho = rho / np.sum(rho)

    Z = np.sum(prob_mat * rho, axis=1)
    weighted_prob_mat = rho[:,None] * prob_mat.T / Z

    grad[:,0] = np.sum((images[:,0] - path[:,0][:,None]) * weighted_prob_mat, axis=1)
    grad[:,1] = np.sum((images[:,1] - path[:,1][:,None]) * weighted_prob_mat, axis=1)


    return -1 / sigma**2 * grad

def numerical_test(path, images, free_energy, sigma=1, dx=0.0001, index=0, n_models=0):
    
    path_test = np.copy(path)
    
    # Calc everything at x = x
    prob_mat = post_prob(path_test, images, sigma)
    log_post1 = neglogpost_cryobife(free_energy, 1, prob_mat)
    grad1 = gradient_cryo_bife(path_test, free_energy, images, prob_mat, 1)

    path_test[n_models, index] += dx

    prob_mat = post_prob(path_test, images, 1)
    log_post2 = neglogpost_cryobife(free_energy, 1, prob_mat)
    grad2 = gradient_cryo_bife(path_test, free_energy, images, prob_mat, 1)

    analt_grad = grad2[n_models, index]
    num_grad = (log_post2 - log_post1)/dx
    
    print(f"Gradient calculated for index {index} of n_models {n_models}")
    print(f"Numerical gradient: {num_grad}")
    print(f"Analytical gradient: {analt_grad}")

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

    th1 = np.arctan2(new_path[1:-1,1] - new_path[:-2,1], new_path[1:-1,0] - new_path[:-2,0])
    th2 = np.arctan2(new_path[2:,1] - new_path[1:-1,1], new_path[2:,0] - new_path[1:-1,0])

    der = np.sum((th1 - th2)**2)

    return der

##NUMBA_THREADING_LAYER='omp'
##config.THREADING_LAYER = 'threadsafe'
##@njit(parallel=True)
##@jit
def langevin(path, images, lmc_steps):

    kappa = 1
    sigma = 0.5

    kappa_2 = 90
    alpha = 60
    beta = 1.0
    h = 0.001

    #lmc_steps = 10
    num_rejected = 0

    G_acc = []
    mala_acc = []
    log_post_acc = []
    path_x_coords_acc = []
    path_y_coords_acc = []

    old_path = np.copy(path)
    n_models = old_path.shape[0]
    index = old_path.shape[1]

    mala_num = -9999999999999

    for step in range(lmc_steps):

        print(f'LMC step {step}')

        new_path = np.copy(old_path)

        value = False
        while value == False:

            model_index = np.random.randint(0,n_models)

            if(model_index==0 or model_index==7 or model_index==13):

                print(f'Node {model_index} is fixed...choosing different node')
                continue

            else: 
                
                value = True

        coord_index = np.random.randint(0, 2)
        print(f'n_models = {model_index} ; coord = {coord_index}')

        Xi = np.random.randn() #(n_models, index)
        G_rand = 1.0 * np.random.randn(n_models)
        prob_mat = post_prob(old_path, images, sigma)
        G_op = so.minimize(neglogpost_cryobife, G_rand ,method='CG',args=(kappa,prob_mat,))
        log_post_old = -1*G_op.fun
        free_energy_old = G_op.x

        gradient_old = gradient_cryo_bife(old_path, free_energy_old, images, prob_mat, sigma)
        new_path[model_index,coord_index] += -h*gradient_old[model_index,coord_index] + np.sqrt(2*h)*Xi #[model_index,coord_index]

        der = models_der(new_path)
        dist = models_dist(new_path)
  
        MALA_den = -log_post_old - np.sum( (new_path - old_path + h * gradient_old)**2 ) - kappa_2*dist - alpha*der

        aa=0
        rr = np.log(np.random.random())

        if (MALA_den > mala_num):

            aa += 1
            old_path = np.copy(new_path)
            mala_num = MALA_den
            mala_acc.append(MALA_den)

            prob_mat_new = post_prob(new_path, images, sigma)
            G_op_new = so.minimize(neglogpost_cryobife, G_rand ,method='CG',args=(kappa,prob_mat_new,))
            log_post_new = -1*G_op_new.fun
            free_energy_new = G_op_new.x

            path_x_coords_acc.append(new_path[:,0])
            path_y_coords_acc.append(new_path[:,1])
            G_acc.append(free_energy_new)
            log_post_acc.append(log_post_new)

            print('Path Accepted ')

        elif (rr < -(mala_num-MALA_den)*beta):

            aa += 2
            old_path = np.copy(new_path)
            mala_num = MALA_den
            mala_acc.append(MALA_den)

            prob_mat_new = post_prob(new_path, images, sigma)
            G_op_new = so.minimize(neglogpost_cryobife, G_rand ,method='CG',args=(kappa,prob_mat_new,))
            log_post_new = -1*G_op_new.fun
            free_energy_new = G_op_new.x

            path_x_coords_acc.append(new_path[:,0])
            path_y_coords_acc.append(new_path[:,1])
            G_acc.append(free_energy_new)
            log_post_acc.append(log_post_new)

            print('Path Accepted ')

        else:

            old_path = old_path
            MALA_den = MALA_den
            num_rejected += 1
            print('Path Rejected')

        print('ProbsPilar', aa, MALA_den,mala_num, -dist, -der,log_post_new)

        #print(f'Valor Neg_logpost = {-log_post}')
        #print(f'Valor MALA_New = {MALA_New}') 

        """
        plt.figure()
        plt.plot([i for i in range(len(free_energy))],free_energy,'-o', label = 'free energy')
        plt.legend()
        plt.show()
        """
    G_acc = np.array(G_acc)
    mala_acc = np.array(mala_acc)
    log_post_acc = np.array(log_post_acc)
    path_x_coords_acc = np.array(path_x_coords_acc)
    path_y_coords_acc = np.array(path_y_coords_acc)

    np.savetxt('G_acc',G_acc)
    np.savetxt('mala_acc',mala_acc)
    np.savetxt('log_post_acc',log_post_acc)
    np.savetxt('path_x_coords_acc',path_x_coords_acc)
    np.savetxt('path_y_coords_acc',path_y_coords_acc)
    
    print('Total rejected paths =',num_rejected)

    return(new_path)

def bife_fes_opt(path, images, sigma):

    kappa = 1

    n_models = path.shape[0]

    G_rand = 1.0 * np.random.randn(n_models)
    prob_mat = post_prob(path, images, sigma)

    G_op = so.minimize(neglogpost_cryobife, G_rand, method='CG', args=(kappa, prob_mat))

    return G_op

def do_langevin(initial_path, images, G, steps):

    sigma = 0.5

    kappa_2 = 9*1e2
    alpha = 6*1e2
    beta = 1.0
    h = 0.0001

    n_models = initial_path.shape[0]

    # Variables related to G
    new_log_post = -1 * G.fun
    free_energy = G.x

    # Calculate "old" variables
    old_path = initial_path.copy()
    old_prob_mat = post_prob(old_path, images, sigma)
    old_gradient = gradient_cryo_bife(old_path, free_energy, images, old_prob_mat, sigma)
    old_neg_post = neglogpost_cryobife(free_energy, kappa_2, old_prob_mat)

    old_der = models_der(old_path)
    old_dist = models_dist(old_path)

    for step in range(steps):

        new_path = old_path.copy()

        # Selecting which replica to update

        value = True
        while value:

            model_index = np.random.randint(0, n_models)
            if(model_index==0 or model_index==7 or model_index==13):

                #print(f'Node {model_index} is fixed...choosing different node')
                continue

            else: value = False

        coord_index = np.random.randint(0, 2)

        # random noise for Langevin
        xi = np.random.randn()

        # Calcualte new proposal
        new_path[model_index, coord_index] += -h * old_gradient[model_index, coord_index] + np.sqrt(2 * h) * xi

        # Not sure what these are for
        mala_den = old_neg_post - np.sum((new_path - old_path + h * old_gradient)**2 / (4*h), axis=1) - kappa_2 * old_dist - alpha * old_der

        new_prob_mat = post_prob(new_path, images, sigma)
        new_neg_post = neglogpost_cryobife(free_energy, kappa_2, new_prob_mat)
        new_gradient = gradient_cryo_bife(new_path, free_energy, images, new_prob_mat, sigma)

        # Not sure what these are for
        new_der = models_der(new_path)
        new_dist = models_dist(new_path)
        
        # !TODO ask Julian where did this come from
        mala_num = new_neg_post - np.sum((old_path - new_path + h * new_gradient)**2 / (4*h), axis=1) - kappa_2 * new_dist - alpha * new_der

        aa = 0
        rr = np.log(np.random.random())

        if (mala_den[model_index] > mala_num[model_index]):

            aa += 1

            # Update old variables
            old_path = new_path.copy()
            
            old_prob_mat = new_prob_mat.copy()
            old_gradient = new_gradient.copy()
            old_neg_post = new_neg_post

            old_dist = new_dist
            old_der = new_der

        elif (rr < -(mala_num[model_index] - mala_den[model_index]) * beta):

            # why?
            aa += 2

            # Update old variables
            old_path = new_path.copy()
            
            old_prob_mat = post_prob(old_path, images, sigma)
            old_gradient = gradient_cryo_bife(old_path, free_energy, images, old_prob_mat, sigma)

            old_prob_mat = new_prob_mat.copy()
            old_gradient = new_gradient.copy()
            old_neg_post = new_neg_post

            old_dist = new_dist
            old_der = new_der


        else:

            continue
    
    # returns last accepted path
    return old_path

def do_langevin_full(initial_path, images, G, steps):

    sigma = 0.5

    kappa_2 = 9*1e2*0
    alpha = 6*1e2*0
    beta = 1.0
    h = 0.0001

    n_models = initial_path.shape[0]
    n_dims = initial_path.shape[1]

    # Variables related to G
    new_log_post = -1 * G.fun
    free_energy = G.x

    # Calculate "old" variables
    old_path = initial_path.copy()
    old_prob_mat = post_prob(old_path, images, sigma)
    old_gradient = gradient_cryo_bife(old_path, free_energy, images, old_prob_mat, sigma)
    old_neg_post = neglogpost_cryobife(free_energy, kappa_2, old_prob_mat)

    old_der = models_der(old_path)
    old_dist = models_dist(old_path)

    mask = np.ones_like(old_path)

    mask[0] = np.zeros((2,))
    mask[7] = np.zeros((2,))
    mask[13] = np.zeros((2,))

    for step in range(steps):

        new_path = old_path.copy()

        # Selecting which replica to update

        # random noise for Langevin
        xi = np.random.randn(n_models, n_dims)

        # Calcualte new proposal
        new_path += (-h * old_gradient + np.sqrt(2 * h) * xi) * mask

        # Not sure what these are for
        mala_den = old_neg_post - np.sum((new_path - old_path + h * old_gradient)**2 / (4*h), axis=1) - kappa_2 * old_dist - alpha * old_der

        new_prob_mat = post_prob(new_path, images, sigma)
        new_neg_post = neglogpost_cryobife(free_energy, kappa_2, new_prob_mat)
        new_gradient = gradient_cryo_bife(new_path, free_energy, images, new_prob_mat, sigma)

        # Not sure what these are for
        new_der = models_der(new_path)
        new_dist = models_dist(new_path)
        
        # !TODO ask Julian where did this come from
        mala_num = new_neg_post - np.sum((old_path - new_path + h * new_gradient)**2 / (4*h), axis=1) - kappa_2 * new_dist - alpha * new_der

        rr = np.log(np.random.rand(*mala_den.shape))

        accp = rr < -(mala_num - mala_den) * beta
        accp = np.array([accp, accp]).T

        old_path = ~accp * old_path + accp * new_path

        if ~accp.all(): 
            
            continue

        else:  

            old_prob_mat = post_prob(old_path, images, sigma)
            old_gradient = gradient_cryo_bife(old_path, free_energy, images, old_prob_mat, sigma)
            old_neg_post = neglogpost_cryobife(free_energy, kappa_2, old_prob_mat)

            old_der = models_der(old_path)
            old_dist = models_dist(old_path)

            
    
    # returns last accepted path
    return old_path

def path_optimization(iterations, initial_path, images, mala_steps):

    sigma = 0.5

    paths = np.zeros((iterations+1, *initial_path.shape))
    paths[0] = initial_path

    curr_path = initial_path.copy()

    for it in range(iterations):

        G = bife_fes_opt(curr_path, images, sigma)
        curr_path = do_langevin(curr_path, images, G, mala_steps)

        paths[it+1] = curr_path
    
    np.save("paths.npy", paths)

    return 0

def main():

    LMC_steps = 100
    path = np.loadtxt("data/Orange")
    images = np.loadtxt("data/images.txt")
    images = images - 10*np.ones(images.shape)
    path_test = np.copy(path) - 11*np.ones(path.shape)

    
    start_time = time.time()
    path_optimization(20, path_test, images, LMC_steps)
    #langevin(path_test,images,LMC_steps)

    print(f"The program takes {(time.time() - start_time):.4f} seconds for {LMC_steps} LMC steps---")
    
    return 0

if __name__ == "__main__":
    
    main()
