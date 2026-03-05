from mac.optimization.frankwolfe import frank_wolfe
from mac.optimization.constraints import solve_subset_box_lp
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2002)

n = 100 # x \in R^n
K = 62 # 1'x = K

A = np.random.random((n, n))
A = .5 *(A@A.T)


def problem(x):
    f = -0.5*x.T @ A @ x
    gradf = -A @ x
    # gradf = x.T@A
    
    return (f,gradf)

X = np.ones(n) * K/n
solve_lp = lambda g: solve_subset_box_lp(g,K)
max_iters=100
relative_duality_gap_tol=1e-5
min_relative_cost_reduction=1e-8
grad_norm_tol=1e-8
verbose=True
log={"iterations": {"iteration": [], "time": [], "point": [], "cost": [], "gradient_norm": [], "dual":[]}}

sol, u = frank_wolfe(initial=X, problem=problem, 
                  solve_lp=solve_lp, maxiter=max_iters,
                    relative_duality_gap_tol=relative_duality_gap_tol,
                    grad_norm_tol=grad_norm_tol,
                    min_relative_cost_reduction_tol=min_relative_cost_reduction,
                    verbose=verbose,log=log)
iterations = log["iterations"]["iteration"]
cost = np.array(log["iterations"]["cost"])
dual = np.array(log["iterations"]["dual"])
dual[0] = np.nan

duality_gap = -(dual[1:]-cost[1:])/cost[1:]
breakpoint()
plt.figure()
plt.plot(iterations[1:], duality_gap, marker='x')
plt.yscale('log')



plt.figure()
plt.plot(iterations, cost, marker='o')
plt.plot(iterations, dual, marker='x')
plt.legend(["cost", "dual"])
plt.show()
print("iterations:", log["iterations"]["iteration"])
print("Cost:", sol.T @ A @ sol) 