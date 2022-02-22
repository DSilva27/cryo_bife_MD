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

def gradient_cryo_bife(path, G, images, prob_mat, sigma, beta=1):

    grad = np.zeros(path.shape)

    rho = np.exp(-beta * G)
    rho = rho / np.sum(rho)

    Z = np.sum(prob_mat * rho, axis=1)
    weighted_prob_mat = rho[:,None] * prob_mat.T / Z

    grad[:,0] = np.sum((images[:,0] - path[:,0][:,None]) * weighted_prob_mat, axis=1)
    grad[:,1] = np.sum((images[:,1] - path[:,1][:,None]) * weighted_prob_mat, axis=1)


    return -1 / sigma**2 * grad

def numerical_test(path, images, free_energy, sigma=1, dx=0.0001, index=0, model=0):
    
    path_test = np.copy(path)
    
    # Calc everything at x = x
    prob_mat = post_prob(path_test, images, sigma)
    log_post1 = neglogpost_cryobife(free_energy, 1, prob_mat)
    grad1 = gradient_cryo_bife(path_test, free_energy, images, prob_mat, 1)

    path_test[model, index] += dx

    prob_mat = post_prob(path_test, images, 1)
    log_post2 = neglogpost_cryobife(free_energy, 1, prob_mat)
    grad2 = gradient_cryo_bife(path_test, free_energy, images, prob_mat, 1)

    analt_grad = grad2[model, index]
    num_grad = (log_post2 - log_post1)/dx
    
    print(f"Gradient calculated for index {index} of model {model}")
    print(f"Numerical gradient: {num_grad}")
    print(f"Analytical gradient: {analt_grad}")

def main():
    
    path = np.loadtxt("data/path.txt")
    free_energy = np.loadtxt("data/free_energy.txt")
    gauss_image_vector = np.loadtxt("data/images.txt")
    
    numerical_test(path, gauss_image_vector, free_energy, sigma=1, dx=0.0001, index=0, model=0)
    
    return 0

if __name__ == "__main__":
    
    main()
