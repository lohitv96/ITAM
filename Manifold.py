import autograd.numpy as np
from pymanopt.manifolds import Stiefel, Euclidean
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent


def cost(X):
    return np.sum(X)


manifold = Stiefel(5, 2)
problem = Problem(manifold=manifold, cost=cost)
solver = SteepestDescent()

Xopt = solver.solve(problem)
print(Xopt)
