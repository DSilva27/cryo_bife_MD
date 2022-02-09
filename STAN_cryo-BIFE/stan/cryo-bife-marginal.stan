/**
 * Model of biomolecule conformational variability and underlying
 * free-energy landscape.  This is roughly the same model as found in
 * cryo-bife.stan, but with an improper prior on the random walk scale 
 * that enables it to be easily marginalized out of the log density.

 * The data is derived from a set of N cryo-EM
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
 * by pinning G[M] = 0.  

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
 * by pinning G[M] = 0.  Rather than assigning the precision lambda
 * prior as in cryo-bife.stan, here it is assigned an impropr uniform
 * prior
 *
 *     lambda ~ uniform(0, inf)
 *
 * and then marginalized out as shown in Giraldo-Barreto et al. (2021)
 * to yield the density as implemented in Stan code.  The prior
 * factors to 
 *
 * log p(G) = log INTEGRAL_0^inf log p(G, lambda)
 *          = log(1 / mathcalG^2)
 *          = -2 log(mathcalG)
 *
 * where mathcalG = sum(diff(G)^2) and diff(G) = G[2:N] - G[1:N - 1].
 *
 * As with the cryo-bife.stan model, this code could be extended to
 * calculate the posterior probability distribution over states for
 * the images.
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
  // return the lag 1 differences between the last N-1 and first N-1
  // elements of the specified N-vector
  vector diff(vector a) {
    int N = rows(a);
    return a[2:N] - a[1:N - 1];
  }
}
data {
  int<lower=0> M;
  int<lower=0> N;
  matrix<lower=0, upper=1>[N, M] Pmat;
}
parameters {
  vector[M - 1] G;
}
transformed parameters {
  simplex[M] rho = softmax(-append_row(G, 0));
}
model {
  // prior
  target += -2 * log(sum(diff(G)^2));

  // likelihood
  target += sum(log(Pmat * rho));
}
