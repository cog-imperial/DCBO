import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gpflow as gpf
from gpflow.config import default_float

gpf.config.set_default_float(tf.float64)
from tensorflow_probability import bijectors as tfb


# set the bounds and initial values for hyperparameters
def bounded_hyperparameter(low, high, lengthscale):
    sigmoid = tfb.Sigmoid(tf.cast(low, tf.float64), tf.cast(high, tf.float64))
    parameter = gpf.Parameter(lengthscale, transform=sigmoid, dtype=tf.float64)
    return parameter


# GPR model via GPflow, refer to https://gpflow.github.io/GPflow/2.5.2/notebooks/basics/regression.html
def GPR(data):
    dim = data[0].shape[1]  # input dimension
    model = gpf.models.GPR(data, kernel=gpf.kernels.Matern52(lengthscales=np.ones(dim)))

    # initialize hyperparameters
    model.kernel.lengthscales = bounded_hyperparameter(
        5e-3 * np.ones(dim), 2.0 * np.ones(dim), 0.5 * np.ones(dim)
    )
    model.kernel.variance = bounded_hyperparameter(5e-2, 20.0, 1)
    model.likelihood.variance = bounded_hyperparameter(5e-4, 0.2, 5e-3)

    # optimize hyperparameters via maximizing loglikelihood
    opt = gpf.optimizers.Scipy()
    opt.minimize(
        model.training_loss, model.trainable_variables, options={"maxiter": 2000}
    )

    return model


# Under independence assumption, MOGP consists of p independent GPs
def Independent_MOGP(data):
    P = data[1].shape[1]  # number of constraints
    model = [GPR((data[0], tf.expand_dims(data[1][:, _], axis=1))) for _ in range(P)]
    return model


# MOGP model via GPflow, refer to https://gpflow.github.io/GPflow/2.5.2/notebooks/advanced/multioutput.html
def Dependent_MOGP(data):
    dim = data[0].shape[1]  # input dimension
    P = data[1].shape[1]  # number of constraints
    L = P  # number of latent independent GPs, we use a full representation here. Set L < P for a low-dimensional representation

    # create kernel list for each latent GPs
    kern_list = [gpf.kernels.Matern52(lengthscales=np.ones(dim)) for _ in range(L)]
    # create multi-output kernel from kernel list
    # initialise the mixing matrix W
    kernel = gpf.kernels.LinearCoregionalization(kern_list, W=np.random.randn(P, L))

    # initialisation of inducing input locations (M random points from the training inputs)
    Z = data[0]
    M = Z.shape[
        0
    ]  # we use all training inputs here. For a sparse version, set a smaller M
    # create multi-output inducing variables from Z
    iv = gpf.inducing_variables.SharedIndependentInducingVariables(
        gpf.inducing_variables.InducingPoints(Z)
    )

    # initialize mean of variational posterior to be of shape MxL
    q_mu = np.zeros((M, L))
    # initialize \sqrt(Σ) of variational posterior to be of shape LxMxM
    q_sqrt = np.repeat(np.eye(M)[None, ...], L, axis=0) * 1.0
    # create SVGP model as usual and optimize
    model = gpf.models.SVGP(
        kernel,
        gpf.likelihoods.Gaussian(),
        inducing_variable=iv,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
    )

    # initialize hyperparameters
    for i in range(L):
        model.kernel.kernels[i].lengthscales = bounded_hyperparameter(
            5e-3 * np.ones(dim), 2.0 * np.ones(dim), 0.5 * np.ones(dim)
        )
        model.kernel.kernels[i].variance = bounded_hyperparameter(5e-2, 20.0, 1)
    model.likelihood.variance = bounded_hyperparameter(5e-4, 0.2, 5e-3)

    # optimize hyperparameters via maximizing loglikelihood
    opt = gpf.optimizers.Scipy()
    opt.minimize(
        model.training_loss_closure(data),
        variables=model.trainable_variables,
        method="l-bfgs-b",
        options={"maxiter": MAXITER},
    )

    return model
