import numpy as np
from sklearn.covariance import log_likelihood

def integrated_prior(G):
    
    mathcal_G = sum(np.diff(G)**2)
    log_prior = np.log(1/mathcal_G**2)    # note kappa scales *log* prior
    
    return log_prior

def post_prob(path, gauss_image_vector, sigma):

    number_of_nodes = path.shape[0]
    number_of_images = gauss_image_vector.shape[0]
    prob_matrix = np.zeros((number_of_images, number_of_nodes))

    norm = (1 / (2 * np.pi * sigma**2))
    
    prob_matrix = norm * np.exp(-((path[:,0][:,None] - gauss_image_vector[:,0])**2 \
                                    + (path[:,1][:,None] - gauss_image_vector[:,1])**2) \
                                      /  (2 * sigma**2))

    return prob_matrix.T

def neglogpost_cryobife(G, kappa, Pmat, log_prior_fxn=None):

    if log_prior_fxn is None: # Default is the prior from the paper

        log_prior_fxn = integrated_prior

    log_prior = kappa * log_prior_fxn(G)
    
    rho = np.exp(-G) #density vec
    rho = rho / np.sum(rho) #normalize, Eq.(8)

    log_likelihood = np.sum(np.log(np.dot(Pmat, rho))) # sum here since iid images; logsumexp
    neg_log_posterior = -(log_likelihood + log_prior)

    return neg_log_posterior # check log_prior sign error?
