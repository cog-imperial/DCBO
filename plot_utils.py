import numpy as np
import matplotlib.pyplot as plt

# plot the samples on the contour of objective
def plot_sample(fun, X_sample, step = 0.001):
    rx, ry = np.arange(0, 1, step), np.arange(0, 1, step)
    gx, gy = np.meshgrid(rx, ry)
    X = np.c_[gx.ravel(), gy.ravel()]
    Objective = fun(X, 0)
    for i in range(1, fun.p+1):
        Constraint = fun(X, i)
        Objective[Constraint >= 0] = np.nan
    Objective = Objective.reshape(gx.shape)
    plt.figure(figsize=(6,4))
    plt.title("Sampling points")
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.contourf(gx, gy, Objective, 200, cmap='rainbow')
    plt.plot(X_sample[:,0], X_sample[:,1], 'kx', mew=3, label='Samples')
    plt.colorbar() 
    plt.show()

# plot the best objective value curve
# for single method, we set the objective value of infeasible samples be Max + 0.1 * (Max - Min), where Min/Max is the minimal/maximal objective value for feasible samples
def plot_answer(Y, index, label, opt):
    Min = min(Y[index == True])
    Max = max(Y[index == True])
    Y[index == False] = Max + 0.1 * (Max - Min)
    Y = np.minimum.accumulate(Y.ravel())
    budget = Y.shape[0]
    x = np.array(range(1, budget + 1))
    plt.figure(figsize=(6, 4))
    plt.xlim(1, budget)
    plt.plot(x, Y, label=label)
    plt.hlines(opt, 1, budget, color='k', linestyle='dotted', label='Optimal Value')  
    plt.legend()
    plt.show()