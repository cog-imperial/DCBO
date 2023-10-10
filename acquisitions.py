import math
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gpflow as gpf
from gpflow.config import default_float
gpf.config.set_default_float(tf.float64)

# constrained expected improvement (cEI)
class constrained_expected_improvement():
    def __init__(self, model, mean_sample, index_sample):
        self._model = model # surrogate model
        if np.count_nonzero(index_sample):
            # if there has at least one feasible point, set flag = False, and let eta be the minimal value for feasible points
            self._flag = False
            self._eta = tf.reduce_min(mean_sample[index_sample == True], axis=0)
        else:
            # if there is no feasible point, set flag = Ture
            self._flag = True

    #given the probability of feasibility (PoF) and return cEI at given point x
    @tf.function
    def __call__(self, x, pof):
        if self._flag:
            # if there is no feasible point, set EI term be 1. In this case, cEI reduces to PoF
            return pof
        mean, variance = self._model.predict_f(x)
        # otherwise, calculate expected improvement (EI), and return the product of EI and PoF
        normal = tfp.distributions.Normal(mean, tf.sqrt(variance))
        return pof * ((self._eta - mean) * normal.cdf(self._eta) + variance * normal.prob(self._eta))

# constrained adaptive sampling (cAS)
class constrained_adaptive_sampling():
    def __init__(self, model, X_sample, mean_sample, index_sample):
        self._model = model # surrogate model
        self._X = X_sample 
        self._w = np.array([10., 1., 1.]) # weight
        if np.count_nonzero(index_sample):
            # if there has at least one feasible point, set flag = False, and let min/max be the minimal/maximal value for feasible points
            self._flag = False
            self._min = tf.reshape(tf.math.reduce_min(mean_sample[index_sample == True]), [1, 1])
            self._max = tf.reshape(tf.math.reduce_max(mean_sample[index_sample == True]), [1, 1])
        else:
            # if there is no feasible point, set flag = Ture
            self._flag = True
        gamma0 = 1
        d = 0.25
        N = X_sample.shape[0] # number of samples
        self._gamma = gamma0 / (d ** 2) * np.log(1. - np.power(0.5, N)) # gamma

    #given the probability of feasibility (PoF) as constraint term and return cAS at given point x
    @tf.function
    def __call__(self, x, pof):
        if self._flag:
            # if there is no feasible point, set optimization term be 0. In this case, cAS consists of exploration term and constraint term
            Uo = tf.constant([[0.]], dtype = default_float())
        else:
            # otherwise, calculate optimization term
            mean, variance = self._model.predict_f(x)
            if tf.greater(mean, self._max):
                Uo = tf.constant([[0.]], dtype = default_float())
            elif tf.greater(self._min, mean):
                Uo = tf.constant([[1.]], dtype = default_float())
            else:
                Uo = (self._max - mean) / (self._max - self._min)
        # calculate exploration term
        Ur = tf.constant([[1.]], dtype = default_float())
        for x0 in self._X:
            Ur = Ur * (1. - tf.math.exp(self._gamma * (tf.norm(x - x0) ** 2)))
        # return the weight sum of three terms
        return (self._w[0] * Uo + self._w[1] * pof + self._w[2] * Ur) / np.sum(self._w) 
    
# probability of feasibility (PoF)
# calculate the product of P(c_i(x) <= tolerance) for some tolerance
class independent_probability_of_feasibility():
    def __init__(self, model, tolerance = 0):
        self._model = model
        self._tolerance = tolerance
        
    # return PoF for given point x
    @tf.function
    def __call__(self, x):
        pof = tf.ones((1, ), dtype=default_float())
        for model in self._model:
            mean, variance = model.predict_f(x)
            mean = mean - self._tolerance
            normal = tfp.distributions.Normal(mean, tf.sqrt(variance))
            pof = pof * normal.cdf(tf.cast(0, x.dtype))
        return pof

# Expectation propagation for multivariate gaussian probabilities (EPMGP)
# The original Matlab code could be found via https://github.com/cunni/epmgp provided by Cunningham et al.  
# Refer to Cunningham et al. http://arxiv.org/abs/1111.6832 for details
# Here we only implement the cdf case in Python. Given m, K, return the probability of P(x <= 0), where x ~ N(m, K)
@tf.function
def EPMGP(m, K):
    n = K.shape[0]
    if tf.math.equal(tf.linalg.det(K), 0):
        K = K + tf.eye(n, dtype=tf.float64) * 1e-8
    eps = 1e-3
    mu = tf.fill([n, 1], tf.constant(-100, dtype=tf.float64))
    Sigma = K
    Kinvm = tf.linalg.solve(K, m)

    tauSite = tf.zeros([n, 1], dtype=tf.float64)
    nuSite = tf.zeros([n, 1], dtype=tf.float64)

    muLast = tf.fill([n, 1], tf.constant(-np.inf, dtype=tf.float64))

    for i in tf.range(25):

        Sigma_diag = tf.expand_dims(tf.linalg.diag_part(Sigma), axis=1)
        tauCavity = 1 / Sigma_diag - tauSite
        nuCavity = mu / Sigma_diag - nuSite

        mu_in = nuCavity / tauCavity
        sigma_in = 1 / tauCavity

        logZhat = tf.zeros([n, 1], dtype=tf.float64)
        muhat = tf.zeros([n, 1], dtype=tf.float64)
        sighat = tf.zeros([n, 1], dtype=tf.float64)

        for i in range(n):
            mu_temp = mu_in[i, 0]
            sigma_temp = sigma_in[i, 0]
            b = - mu_temp / tf.sqrt(2.0 * sigma_temp)

            if tf.math.greater(b, 26):
                logZhatOtherTail =  tf.math.log(tf.constant(0.5, dtype=tf.float64)) + tf.math.log(tfp.math.erfcx(b)) - tf.square(b)
                logZhat_temp = tf.math.log1p(-tf.math.exp(logZhatOtherTail))
            else:
                logZhat_temp = tf.math.log(tf.constant(0.5, dtype=tf.float64)) + tf.math.log(tfp.math.erfcx(-b)) - tf.square(b)

            mean_const = - 2.0 / tfp.math.erfcx(-b)
            var_const = - 2.0 / tfp.math.erfcx(-b) * mu_temp

            muhat_temp = mu_temp + mean_const * tf.sqrt(sigma_temp / (2.0 * math.pi))
            sighat_temp = sigma_temp + var_const * tf.sqrt(sigma_temp / (2.0 * math.pi)) + tf.square(mu_temp) - tf.square(muhat_temp)

            logZhat = tf.concat([logZhat[:i], [[logZhat_temp]] , logZhat[i+1:]], axis=0)
            muhat = tf.concat([muhat[:i], [[muhat_temp]] , muhat[i+1:]], axis=0)
            sighat = tf.concat([sighat[:i], [[sighat_temp]] , sighat[i+1:]], axis=0)

        deltatauSite = 1 / sighat - tauCavity - tauSite
        tauSite_temp = tauSite + deltatauSite

        tauSite = tf.math.maximum(tauSite_temp, 0)
        nuSite = muhat / sighat - nuCavity

        SsiteHalf = tf.linalg.tensor_diag(tf.squeeze(tf.sqrt(tauSite)))
        L = tf.linalg.cholesky(tf.eye(n, dtype=tf.float64) + tf.matmul(tf.matmul(SsiteHalf, K), SsiteHalf))
        V = tf.linalg.solve(L, tf.matmul(SsiteHalf, K))
        Sigma = K - tf.matmul(tf.linalg.matrix_transpose(V), V)
        mu = tf.matmul(Sigma, nuSite + Kinvm)

        if tf.norm(muLast-mu) < eps:
            break
        else:
            muLast = mu

    lZ1 = 0.5 * tf.reduce_sum(tf.math.log(1 + tauSite / tauCavity)) - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(tf.squeeze(L))))
    lZ2 = 0.5 * tf.matmul(tf.matmul(tf.linalg.matrix_transpose(nuSite - tauSite * m), Sigma - tf.linalg.tensor_diag(tf.squeeze(1/(tauCavity + tauSite)))), nuSite - tauSite * m)
    lZ3 = 0.5 * tf.matmul(tf.linalg.matrix_transpose(nuCavity), tf.linalg.solve(tf.linalg.tensor_diag(tf.squeeze(tauSite)) + tf.linalg.tensor_diag(tf.squeeze(tauCavity)), tauSite * nuCavity / tauCavity -  2 * nuSite))
    lZ4 = -0.5 * tf.matmul(tf.linalg.matrix_transpose(tauCavity * m), tf.linalg.solve(tf.linalg.tensor_diag(tf.squeeze(tauSite)) + tf.linalg.tensor_diag(tf.squeeze(tauCavity)), tauSite * m - 2 * nuSite))
    logZEP = lZ1 + lZ2 + lZ3 + lZ4 + tf.reduce_sum(logZhat)

    return tf.math.exp(tf.math.minimum(logZEP, 0))

# probability of feasibility with dependence (Dep-PoF)
# calculate P(c_1(x) <= tolerance, ..., c_p(x) <= tolerance) for some tolerance
class dependent_probability_of_feasibility():
    def __init__(self, model, tolerance = 0):
        self._model = model
        self._tolerance = tolerance
    
    @tf.function
    # return Dep-PoF for given point x
    def __call__(self, x):
        m, K = self._model.predict_f(x, full_cov = False, full_output_cov = True)
        m = tf.linalg.matrix_transpose(m) - self._tolerance
        K = K[0]
        # Given mean and covariance matrix, calculate Dep-PoF by EPMGP algorithm
        return EPMGP(m, K)
    
# probability of feasibility using MOGP (Indep-PoF)
class independent_probability_of_feasibility_MOGP():
    def __init__(self, model, tolerance = 0):
        self._model = model
        self._tolerance = tolerance
    
    @tf.function
    # return Dep-PoF for given point x
    def __call__(self, x):
        m, K = self._model.predict_f(x)
        p = m.shape[1]
        pof = tf.ones((1, ), dtype=default_float())
        for i in range(p):
            mean = m[0, i] - self._tolerance
            variance = K[0, i]
            normal = tfp.distributions.Normal(mean, tf.sqrt(variance))
            pof = pof * normal.cdf(tf.cast(0, x.dtype))
        return pof
        