import numpy as np
import tensorflow as tf
from scipy.optimize import differential_evolution
import gpflow as gpf
import time
from gpflow.config import default_float

gpf.config.set_default_float(tf.float64)
from scipy.stats import qmc

from functions import (
    Gardner,
    Gramacy,
    Sasena,
    G4,
    G6,
    G7,
    G8,
    G9,
    G10,
    Tension_Compression,
    Pressure_Vessel,
    Welded_Beam,
    Speed_Reducer,
)
from acquisitions import constrained_expected_improvement, constrained_adaptive_sampling
from acquisitions import (
    independent_probability_of_feasibility,
    dependent_probability_of_feasibility,
    independent_probability_of_feasibility_MOGP,
)
from plot_utils import plot_sample, plot_answer
from models import GPR, Independent_MOGP, Dependent_MOGP
import sys


# noise of evaluation
noise = 0
# tolerance of error
tolerance = 0

methods = [
    "cEI",
    "cAS",
    "Dep-cEI",
    "Dep-cAS",
    "Indep-cEI",
    "Indep-cAS",
]  # list of methods
funs = [
    Gardner(2, 2),
    Gardner(2, 3),
    Gardner(2, 4),
    Gardner(2, 5),
    Gramacy(),
    Sasena(),
    G4(),
    G6(),
    G7(),
    G8(),
    G9(),
    G10(),
    Tension_Compression(),
    Pressure_Vessel(),
    Welded_Beam(),
    Speed_Reducer(),
]  # list of benchmarks

fun_index = int(sys.argv[1])  # the index of problem in list of benchmarks
method_index = int(sys.argv[2])  # the index of method in list of method
budget = int(sys.argv[3])

assert fun_index in range(16)
assert method_index in range(6)

fun = funs[fun_index]
method = methods[method_index]
print(f"method={method}, fun={fun.name}, budget={budget}")

bounds = np.hstack(
    [np.zeros([fun.dim, 1]), np.ones([fun.dim, 1])]
)  # bounds of variables

# set random seed for each run
seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)


# objective function(index = 0) and constraints(1 <= index <= p)
def f(fun, x, index, noise=noise):
    ans = tf.convert_to_tensor(fun(x.numpy(), index))
    if index:
        ans = np.sign(ans) * np.log(1.0 + np.abs(ans))
        ans = np.sign(ans) * np.log(1.0 + np.abs(ans))
    return tf.expand_dims(ans, axis=1) + noise * tf.random.normal(
        (x.shape[0], 1), dtype=default_float()
    )


# objective value
def objective(fun, x):
    return f(fun, x, 0)


# constraint values
def constraints(fun, x):
    ans = f(fun, x, 1)
    for i in range(2, fun.p + 1):
        ans = tf.concat([ans, f(fun, x, i)], axis=1)
    return ans


# standardize the objective values to 0 mean and 1 variance
def standardize(Y):
    mean = tf.math.reduce_mean(Y, axis=0)
    std = tf.math.reduce_std(Y, axis=0)
    return (Y - mean) / std


# check the feasibility of samples
def feasible_index(Y):
    index = np.empty(shape=(Y.shape[0], 1))
    P = Y.shape[1]
    for i in range(len(Y)):
        index[i] = True
        for j in range(P):
            if Y[i, j] > tolerance:
                index[i] = False
                break
    return index


# use Latin Hypercube Design (LHD) to initially sample points
def LHD(fun, n_init, seed):
    X = tf.convert_to_tensor(qmc.LatinHypercube(d=fun.dim, seed=seed).random(n=n_init))
    return X, [objective(fun, X), constraints(fun, X)]


# propose the next sample by maximizing the acquisition function
def propose_location(acquisition, PoF):
    def acquisition_numpy(x):
        # a numpy version of negative acquisition function as we are going to use scipy.optimize to minimize a numpy function
        x = tf.convert_to_tensor(np.expand_dims(x, axis=0))
        ans = -acquisition(x, PoF(x))
        return ans.numpy()[0]

    # first apply stochatic differential evolution, then use L-BFGS-B to further refine the result
    res = differential_evolution(acquisition_numpy, bounds, polish=True)
    return tf.constant(np.expand_dims(res.x, axis=0))


n_init = budget // 10  # mumber of initial random samples
X_sample, Y_sample = LHD(fun, n_init, seed)  # initial samples
index_sample = feasible_index(Y_sample[1])  # index of feasibility

# one could plot the samples for 2D problems
# plot_sample(fun, X_sample)

# if no feasible point, flag = False; otherwise, flag = True
flag = False

for k in range(budget - n_init):
    # if we've sampled at least one feasible point, update flag = True
    if np.count_nonzero(index_sample):
        flag = True

    T0 = time.time()

    model = []

    if flag:
        # if there is at least one feasible samples, then apply a GPR to model the objective function
        # standardize the objective values before fitting a GPR
        model.append(GPR((X_sample, standardize(Y_sample[0]))))
    else:
        # otherwise, don't need a GPR for objective function when we don't have feasible samples
        model.append(None)

    if method in ["cEI", "cAS"]:  # PoF
        model.append(Independent_MOGP((X_sample, Y_sample[1])))
        PoF = independent_probability_of_feasibility(model[-1], 0)
    elif method in ["Dep-cEI", "Dep-cAS"]:  # Dep-PoF
        model.append(Dependent_MOGP((X_sample, Y_sample[1])))
        PoF = dependent_probability_of_feasibility(model[-1], 0)
    elif method in ["Indep-cEI", "Indep-cAS"]:  # PoF with MODP
        model.append(Dependent_MOGP((X_sample, Y_sample[1])))
        PoF = independent_probability_of_feasibility_MOGP(model[-1], 0)

    T1 = time.time()

    # for noisy cases, use the predictive values as objective values for all samples
    # mean_sample, _ = model[0].predict_f(X_sample)
    # for noise-free cases, use the (standardized) objective values
    mean_sample = standardize(Y_sample[0])

    if method in ["cEI", "Dep-cEI", "Indep-cEI"]:  # cEI
        acquisition = constrained_expected_improvement(
            model[0], mean_sample, index_sample
        )
    elif method in ["cAS", "Dep-cAS", "Indep-cAS"]:  # cAS
        acquisition = constrained_adaptive_sampling(
            model[0], X_sample, mean_sample, index_sample
        )

    # obtain next sample
    X_next = propose_location(acquisition, PoF)
    # add sample to previous samples
    X_sample = tf.concat([X_sample, X_next], axis=0)
    Y_sample[0] = tf.concat([Y_sample[0], objective(fun, X_next)], axis=0)
    Y_sample[1] = tf.concat([Y_sample[1], constraints(fun, X_next)], axis=0)
    # update index of feasibility
    index_sample = feasible_index(Y_sample[1])
    # show the objective value if sampling a feasible point in this iteration
    if index_sample[-1]:
        print("True", Y_sample[0][-1].numpy())

    T2 = time.time()

    print(f"Iteration {k}: Training {T1-T0}s, Optimizing {T2-T1}s")

index_sample = feasible_index(Y_sample[1])  # update index of feasibility
X = X_sample.numpy()
Y = Y_sample[0].numpy()  # objective values for samples
if flag:
    plot_answer(Y, index_sample, method, fun.opt)  # plot the best objective value curve
