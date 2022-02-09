/**
 * Model of biomolecule conformational variability and underlying
 * free-energy landscape.  The data is derived from a set of N cryo-EM
 * images x[n], each of which is assumed to arise from a
 * conformation z[n] in 1:M.  The data is an N x M (right) stochastic
 * matrix with entries 
 *
 *     Pmat[n, m] propto p(image = x[n] | conformation = z[m])
 *
 * The model is a multi-logit regression with intercepts G[m] for
 * conformation m.  Letting z[n] in 1:M be the conformation of
 * observation n, the complete data likelihood is
 * 
 *     p(z[n], x[n] | G) = p(x[n] | z[n]) * p(z[n] | G).
 * 
 * Marginalizing the conformation indicator z yields
 *
 *     p(x[n] | G) = SUM_m p(x[n] | z[n] = m) * p(z[n] = m | G),
 *
 * where our data Pmat[n, m] propto p(x[n] | z = m), and where
 * the second probability is given by a multi-logit regression,
 *
 *     p(m | G) = categorical(m | softmax(G)).
 *  
 * Putting this together on the log scale,
 *
 *     log p(x[n] | G) = log(PMat * softmax(G))
 *
 * The model is completed with a first-order random-walk prior on G
 * with precision lambda innovations,
 *
 *     p(G) = PROD_{m in 2:M} normal(G[m] | G[m - 1], 1 / sqrt(lambda))
 *
 * The additive invariance of G is identified in the conventional way
 * by pinning G[M] = 0.  The precision parameter lambda is given a prior
 *
 *     lambda ~ lognormal(0, 3)
 *
 * with a 95% interval of (.0028, 360), corresponding to a 95% interval for
 * scale of (.053, 19).
 *
 * The model could be extended to calculate the posterior probability
 * distribution over states for the images.
 *
 * AUTHORS
 * Julian Giraldo-Barreto, Pilar Cossio, Alex Barnett, Bob Carpenter
 * 
 * REFERENCES
 * Julian Giraldo-Barreto, Sebastian Ortiz, Erik H. Thiede, Karen
 * Palacio-Rodriguez, Bob Carpenter, Alex H. Barnett & Pilar Cossio. 2021.
 * A Bayesian approach to extracting free-energy profiles from
 * cryo-electron microscopy experiments.  Scientific Reports 11.  
 * http://dx.doi.org/10.1038/s41598-021-92621-1
 *
 * COPYRIGHT
 * Simons Foundation, Julian Giraldo-Barreto
 *
 * RELEASED UNDER LICENSE
 * BSD-3
 */

functions {
  /**
   * Return log density of a first-order random walk on
   * alpha given innovations with precision tau.
   */
  real rw1_lpdf(vector alpha, real tau) {
    int N = rows(alpha);
    return normal_lpdf(alpha[2:N] | alpha[1:N-1], tau^-0.5);
  }

  /**
   * Return the log density of the image probabilities Pmat under the 
   * mixture simplex rho. 
   */
  real image_lpdf(matrix Pmat, vector rho) {
    return sum(log(Pmat * rho));
  }
}
data {
  int<lower=0> M;
  int<lower=0> N;
  matrix<lower=0, upper=1>[N, M] Pmat;
}
parameters {
  vector[M - 1] G;
  real<lower=0> lambda;
}
transformed parameters {
  simplex[M] rho = softmax(-append_row(G, 0));
}
model {
  lambda ~ lognormal(0, 3);
  G ~ rw1(lambda);
  Pmat ~ image(rho);
}
