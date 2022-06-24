"""Provide functions related to Cryo-BIFE"""
import numpy as np
import scipy.optimize as so
import matplotlib.pyplot as plt
from typing import Callable, Tuple


from numba import jit, njit


class CryoBife:
    """CryoBife provides cryo-bife's prior, likelihood, posterior and
    the optimizer as described in 10.1038/s41598-021-92621-1. """

    @staticmethod
    @jit
    def integrated_prior(fe_prof: np.ndarray) -> float:
        """Calculate the value of the prior for the given free-energy profile.

        :param fe_prof: Array with the values of the free-energy profile in each
            node of the path.

        :returns: The value of the prior for the free-energy profile.
        """
        acc_der_fe_prof = sum(np.diff(fe_prof)**2)
        log_prior = np.log(1 / acc_der_fe_prof**2)

        return log_prior

    @staticmethod
    @jit
    def likelihood(
            path: np.ndarray,
            images: np.ndarray,
            Sample_per_nodes: np.ndarray, 
            sigma: float) -> np.ndarray:
        """Calculate cryo-bife's likelihood matrix given a path and a dataset of images

        :param path: Array with the values of the variables at each node of the path.
                     Shape must be (n_models, n_dimensions).
        :param images: Array with all the experimental images.
                       Shape must be (n_images, image_dimensions)
        :param sigma: Overall noise among the images

        :returns: Array with the likelihood of observing each image given each model.
                  Shape will be (n_images, n_models)
        """

#        number_of_nodes = path.shape[0]
#        number_of_images = images.shape[0]
#        prob_matrix = np.zeros((number_of_images, number_of_nodes))

#        norm = 1 / (2 * np.pi * sigma**2)
#        prob_matrix = norm * np.exp(-0.5 * 1/sigma**2 *\
#                                    np.sum((path[:,None] - images)**2, axis=-1)).T

        norm = 1 / (2 * np.pi * sigma**2)
        number_of_images = images.shape[0]
        number_of_nodes = path.shape[0]
        number_of_samples = Sample_per_nodes.shape[1]
        prob_matrix = np.zeros((number_of_nodes,number_of_samples,number_of_images))

        for i in range(number_of_nodes):
            for j in range(number_of_samples):

                prob_matrix[i,j] = norm * np.exp( (-0.5*1/(sigma**2))*\
                                                  ((Sample_per_nodes[i,j,0] - images[:,0])**2+\
                                                   (Sample_per_nodes[i,j,1] - images[:,1])**2))
        
        
#        for i in range(number_of_nodes):
#            for j in range(number_of_samples):
#                for k in range(number_of_images):
#
#                    prob_matrix[i,j,k] = norm * np.exp(-0.5 * 1/sigma**2 *\
#                                                       np.sum((Sample_per_nodes[i,j,:] - images[k,:])**2)).T
        return prob_matrix

    def neg_log_posterior(
            self,
            fe_prof: np.ndarray,
            kappa: float,
            prob_mat: np.ndarray,
            beta: float = 1,
            prior_fxn: Callable = None) -> float:
        """Calculate cryo-bife's negative log-posterior.

        :param fe_prof: Array with the values of the free-energy profile (FEP)
                        in each node of the path.
        :param beta: Temperature.
        :param kappa: Scaling factor for the prior.
        :prior_fxn: Function used to calculate the FEP's prior

        :returns: Value of the negative log-posterior
        """

        if prior_fxn is None:
            # Default is the prior from the paper
            prior_fxn = self.integrated_prior

        # TODO: Think for a better name for rho
        rho = np.exp(-beta * fe_prof) #density vec
        rho = rho / np.sum(rho) #normalize, Eq.(8)

        # Sum here since iid images; logsumexp
        log_likelihood = np.sum(np.log(np.dot(prob_mat.T, rho)))
        log_prior = kappa * prior_fxn(fe_prof)

        neg_log_posterior = -(log_likelihood + log_prior)

        return neg_log_posterior

    @staticmethod
    def grad2(
            path: np.ndarray,
            Sample_per_nodes: np.ndarray,
            images: np.ndarray,
            fe_prof: np.ndarray,
            sigma: float,
            nu: float,
            Cv_differences: np.ndarray,
            prior_fxn: Callable = None) -> np.ndarray:
        """Calculate cryo-bife's negative log-posterior.

        :param path: Array with the values of the variables at each node of the path.
                     Shape must be (n_models, n_dimensions).
        :param fe_prof: Array with the values of the free-energy profile (FEP)
                        in each node of the path.
        :param images: Array with all the experimental images.
                       Shape must be (n_images, image_dimensions)
        :param sigma: TODO.
        :param beta: Temperature.
        :param kappa: Scaling factor for the prior.
        :prior_fxn: Function used to calculate the FEP's prior

        :returns: Value of the negative log-posterior
        """
        G = fe_prof
        number_of_nodes = G.shape[0]
        prob_matrix = CryoBife.likelihood(path, images, Sample_per_nodes, sigma)
        #print('SHAPE_MATRIX',prob_matrix.shape)
        #print('SHAPE_DIFF',Cv_differences.shape)

        Norm = (1/prob_matrix.shape[1])*np.dot(G,np.sum(prob_matrix,axis=1))
        grad = np.zeros((number_of_nodes,2))
        
        for k in range(number_of_nodes):
        
            gradx = 2*nu*np.exp(-G[k]) * ((1/prob_matrix.shape[1])*np.dot(Cv_differences[k,:,0],prob_matrix[k]) - np.mean(prob_matrix,axis=1)[k,:]*np.mean(Cv_differences[k],axis=0)[0])
            grady = 2*nu*np.exp(-G[k]) * ((1/prob_matrix.shape[1])*np.dot(Cv_differences[k,:,1],prob_matrix[k]) - np.mean(prob_matrix,axis=1)[k,:]*np.mean(Cv_differences[k],axis=0)[1])

            gradx = gradx/Norm
            grady= grady/Norm
            #print('SHAPE_GRADX',gradx.shape)

            grad[k,0] = np.sum(gradx) 
            grad[k,1] = np.sum(grady) 
            
        return grad

    def optimizer(
            self,
            path: np.ndarray,
            images: np.ndarray,
            Sample_per_nodes: np.ndarray,
            sigma: float,
            initial_fe_prof: np.ndarray = None) -> np.ndarray:
        """Find the optimal free-energy profile given a path and a dataset of images

        :param path: Array with the values of the variables at each node of the path.
                     Shape must be (n_models, n_dimensions).
        :param images: Array with all the experimental images.
                       Shape must be (n_images, image_dimensions)
        :param sigma: TODO.
        :param fe_prof: Initial guess for the free-energy profile

        :returns: Optimized free-energy profile
        """

        kappa = 1
        n_models = path.shape[0]

        if initial_fe_prof is None:

            initial_fe_prof = 1.0 * np.random.randn(n_models)

        prob_mat = self.likelihood(path, images, Sample_per_nodes, sigma)
        
        prob_mat = np.mean(prob_mat,axis=1)

        optimized_fe_prof = so.minimize(self.neg_log_posterior,
                                        initial_fe_prof,
                                        method='CG',
                                        args=(kappa, prob_mat))

        return (optimized_fe_prof.x,optimized_fe_prof.fun)


images = np.loadtxt('/home/jgiraldob/Files_for_David_test/Cryo-CV/2-wells_images.txt')
ini_path = np.loadtxt('/home/jgiraldob/example_data/2-wells_Orange-path')
first_path = np.copy(ini_path)
sim_path = np.copy(ini_path)

nu = 1000
sigma = 0.5
sampling_steps = 1001
step_size = 0.00001
gradient_steps = 51
number_of_nodes = ini_path.shape[0]

#BATCH

batch_size = int(images.shape[0] * 0.1)
number_of_batches = images.shape[0]//batch_size
residual_batches = images.shape[0]%batch_size

images_shuffled = images.copy()
np.random.shuffle(images_shuffled)

#END BATCH

Paths = []

mask = np.ones_like(ini_path)
mask[0] = np.zeros((2,))
mask[-1] = np.zeros((2,))

CB = CryoBife()

for s in range(gradient_steps):
    
    Cv_differences = []
    Sample_per_nodes = []

    for j in range(number_of_nodes):

        cv_diff = []
        cv_sampling = []

        ini_cv=np.array([sim_path[j,0],sim_path[j,1]])
        new_cv=np.copy(ini_cv)

        for i in range(sampling_steps):

            diff = new_cv - ini_cv
            grad_energy_funct = 2 * nu * diff
            noise = np.array([np.random.randn(),np.random.randn()])
            new_cv = new_cv - grad_energy_funct*step_size + np.sqrt(2*step_size)*noise

            if i >=1:
                if i%1==0:
                    cv_sampling.append(new_cv)
                    cv_diff.append(diff)
        Sample_per_nodes.append(cv_sampling)
        Cv_differences.append(cv_diff)

    Sample_per_nodes = np.array(Sample_per_nodes)
    Cv_differences = np.array(Cv_differences)

    fe_prof, log_posterior = CryoBife.optimizer(CB,sim_path, images, Sample_per_nodes, sigma)
        
    #BATCH

    for i in range(number_of_batches):

        images_batch = images_shuffled[i*batch_size:(i+1)*batch_size]
        Gra = CryoBife.grad2(sim_path, Sample_per_nodes, images_batch, fe_prof, sigma, nu, Cv_differences)
        sim_path += -step_size * Gra * mask

    if residual_batches != 0:

        images_batch = images_shuffled[(number_of_batches-1)*batch_size:]
        Gra = CryoBife.grad2(sim_path, Sample_per_nodes, images_batch, fe_prof, sigma, nu, Cv_differences)
        sim_path += -step_size * Gra * mask
        
    #END BACTH
    
    
    #Gra = CryoBife.grad2(sim_path, Sample_per_nodes, images, fe_prof, sigma, nu, Cv_differences)
    #sim_path += -step_size*Gra*mask
    SP = np.copy(sim_path)
    Paths.append(SP)
    
Paths = np.array(Paths)
np.save('Path_BIFE_CV', Paths)

plt.figure(figsize=(10,9))
#plt.gca().set_facecolor('navy')
#plt.hist2d(images[:,0],images[:,1],bins=(40,40),cmap=plt.cm.jet)

#for i in range(number_of_nodes):
#    plt.plot(Sample_per_nodes[i,:,0],Sample_per_nodes[i,:,1],'o', color = 'darkgreen')

plt.plot(ini_path[:,0],ini_path[:,1],'-o', color='orange')

#for i in range(Paths.shape[0]):
#    if i%4==0:
#        plt.plot(Paths[i,:,0],Paths[i,:,1],'-*',label = 'Path No %s' %i)

plt.plot(Paths[-1,:,0],Paths[-1,:,1],'r-*',label = 'Path No %s' %Paths.shape[0])

#plt.colorbar()
plt.legend()
plt.show()
