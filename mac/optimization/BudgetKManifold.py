import numpy as np
import jax
import jax.numpy as jnp
import functools
from pymanopt.manifolds.manifold import Manifold

def sigmoid_stable(t):
    s = np.zeros_like(t)
    I = t >=0
    s[I] = 1 / (1 + np.exp(-t[I]))
    s[~I] = np.exp(t[~I]) / (1 + np.exp(t[~I]))
    s = np.clip(s, 1e-6, 1 - 1e-6)
    return s

def solve_c_halley(a, k, opts):
    eps = opts['eps']
    maxit = opts['maxit']
    tol_f   = opts['tol_f']   # Default 1e-12
    tol_c   = opts['tol_c']   # Default 1e-12
    stepcap = opts['stepcap'] # default 4.0

    z0  = sigmoid_stable(a)
    f0  = np.sum(z0) - k
    fp0 = np.sum(z0 * (1 - z0))
    c   = -f0 / max(fp0, eps)

    cL = -40
    cU = +40
    fL = np.sum(sigmoid_stable(a + cL)) - k
    fU = np.sum(sigmoid_stable(a + cU)) - k
    if not (fL<=0 and fU>=0):
        while fL >0: cL *=2; fL = np.sum(sigmoid_stable(a + cL)) - k
        while fU <0: cU /=2; fU = np.sum(sigmoid_stable(a + cU)) - k
    for i in range(maxit):
        z = sigmoid_stable(a + c)
        f =sum(z) - k
        if np.abs(f) <= tol_f: break
        zp = z * (1-z)
        f1 = np.sum(zp)
        f2 = np.sum(zp * (1 - 2*z))
        denom = 2*f1*f1-f*f2
        if np.abs(denom) > 1e-18:
            # Halley increment
            step = (2*f*f1) / denom
        else:
            # fallback to Newton
            step = f / max(f1, eps)
        step = max(min(step, stepcap), -stepcap)
        cnew = c - step

        # Keep inside bracket; if not, bisect
        if cnew <= cL or cnew >= cU:
            cnew = 0.5*(cL + cU)
        fnew = np.sum(sigmoid_stable(a + cnew)) - k

        # Update bracket (f is increasing)
        if fnew < 0:
            cL = cnew
            fL = fnew
        else:
            cU = cnew
            fU = fnew

        # Accept only if |f| decreases; else bisect
        if np.abs(fnew) > np.abs(f):
            cnew = 0.5*(cL + cU)
            fnew = np.sum(sigmoid_stable(a + cnew)) - k

        if np.abs(cnew - c) <= tol_c and abs(fnew) <= tol_f:
            c = cnew
            break
        c = cnew
    return c

class BudgetKManifold(Manifold):
    def __init__(self, n, k=1, run_checks=False, eps_tol=1e-8):
        self.n = n
        self.k = k
        self.opts = {'eps': eps_tol,
                     'maxit': 1000,
                     'tol_f': 1e-12,
                     'tol_c': 1e-12,
                     'stepcap': 4.0}

        name = f"Budget K-Manifold with n:{n}, k:{k}"
        self.run_checks = run_checks
        self.eps_tol = eps_tol
        super().__init__(name, n)

    def dist(self, point, tang_a, tang_b):
        raise NotImplementedError("Distance Not yet Implemented")

    def lincomb(self, a1, d1, a2=None, d2=None):
        if a2 is None:
            return a1 * d1
        else:
            return a1*d1 + a2*d2

    def inner_product(self, X, eta, zeta):
        denom = (1 - X) * X
        return np.sum((eta * zeta) / denom)

    def norm(self, X, eta):
        return np.sqrt(self.inner_product(X, eta, eta))

    def typical_dist(self):
        return np.pi / 4

    def euclidean_to_riemannian_gradient(self, X, eta):
        return self.projection(X, eta * X * (1-X))

    def projection(self, X, eta):
        alpha = np.sum(eta) / np.sum(X * (1 - X))
        return eta - alpha * X * (1 - X)

    def projection_euclidean(self, X, eta):
        return eta - np.mean(eta)

    def log(self, P,R):
        # Refer to this diagram https://www.researchgate.net/figure/a-Exponential-and-logarithmic-maps-in-Riemannian-manifolds-b-Set-of-sample-points-in_fig4_271058307
        return R-P


    def retraction(self, X, U, t=1.0):
        return self.logit_retraction_first_order(X, U, t)
    
    def logit_retraction_first_order(self, X, U, t, opts=None):
        if opts is None:
            opts = self.opts
        q = X * (1 - X)
        alpha = np.sum(U) / np.sum(q)
        U = U - alpha * q

        ut  = t * U

        # Free step in natural (logit) coordinates
        eta = np.log(X) - np.log1p(-X)
        a   = eta + ut / (X * (1 - X))

        # Scalar offset to satisfy the sum exactly
        ksum = np.sum(X)
        c  = solve_c_halley(a, ksum, opts)
        #  c    = solve_c_newton(a, ksum, solver_opts)

        # Map back to (0,1)^n
        y = sigmoid_stable(a + c)
        return y

    # def logit_retraction_second_order(self, X, U, t, opts=None):
    #     if opts is None:
    #         opts = self.opts
        

    def transport(self, X1, X2, d):
        # return self.projection(X2, d)
        return self.transport_gpt(X1, X2, d)
    
    def transport_gpt(self, X1, X2, d):
        xi = d / np.sqrt(X1 * (1 - X1))
        # project to tangent at y in θ-space
        sY = 2*np.sqrt(X2 * (1 - X2))
        xi = xi - ((sY.T * xi) / np.maximum(sY.T * sY, self.eps_tol)) * sY
        #  back to x-space
        w  = np.sqrt(X2 * (1 - X2)) * xi
        return w

    def random_point(self):
        X = np.random.rand(self.n)
        return self.project_to_feasible_set(X)

    def random_tangent_vector(self, X):
        eta = np.random.randn(self.n)
        eta = self.projection(X, eta)
        nrm = self.norm(X, eta)
        return eta / nrm

    def zero_vector(self, X):
        return np.zeros(self.n)

    def check_on_manifold(self, X):
        on_manifold = True
        sum_X = np.sum(X)
        if abs(sum_X - self.k) > 1e-4:
            print(f"Error: The vector X is not on the manifold (sums to {sum_X}, not {self.k})")
            on_manifold = False

        if np.any(X < -self.eps_tol):
            print(f"Entries of X less than 0: {X[X < -self.eps_tol]}")
            on_manifold = False

        if np.any(X > 1 + self.eps_tol):
            print(f"Entries of X greater than 1: {X[X - 1 > self.eps_tol]}")
            on_manifold = False
        
        return on_manifold

    def center_pt(self):
        return np.ones(self.n) * self.k / self.n

    # @functools.partial(jax.jit, static_argnums=0)
    def project_to_feasible_set(self, X):
        # return self.dykstras_projection(X)
        return self.theta_shift_projection(X)

    def theta_shift_projection(self, X):
        theta = np.log(X) - np.log1p(X)
        c = solve_c_halley(theta, self.k, self.opts)
        y = sigmoid_stable(theta + c)
        return y

    def dykstras_projection(self, y):
        max_iters = 1000
        # tol = self.eps_tol
        tol = 1e-3
        x = y.copy()
        p = np.zeros(self.n)
        q = np.zeros(self.n)

        for iter in range(max_iters):
            x_prev = x.copy()


            # Project onto the k-simplex (with correction)
            x_tilde = x + p
            x = self.project_to_sum_plane(x_tilde)
            p = x_tilde - x

            # Project onto the hypercube [0, 1]^n (with correction)
            x_tilde = x + q
            x = np.clip(x_tilde, 0 + tol, 1 - tol)
            q = x_tilde - x

            # Check for convergence
            if (np.linalg.norm(x - x_prev) < 1e-8) and (np.max(x) < 1 - tol) and (np.min(x) > 0 + tol):
                break
            # if (np.max(x) <= 1 - tol) and (np.min(x) >= 0 + tol):
            #     break
            if iter == max_iters - 1:
                print("Reached Max iterations")
                print(f"Sum: {np.sum(x)}, Max: {np.max(x)}, Min: {np.min(x)}")

        return x
    
    def project_to_sum_plane(self, pt):
        dist_to_plane = self.k - jnp.sum(pt)
        return pt + dist_to_plane / self.n



if __name__=="__main__":
    import numpy as np
    import pymanopt
    from pymanopt.optimizers import SteepestDescent, ConjugateGradient
    from pymanopt.optimizers.line_search import BackTrackingLineSearcher
    import cProfile
    import pstats
    import cvxpy
    # k = 625
    # n = 1000
    k = 124
    n = 500

    np.random.seed(2002)
    A = np.random.randn(n, n)
    A = A @ A.T

    # A = np.array([[8.8181,   -2.7709,   -0.4774,    2.2081,   -1.5825,   -0.4233,   -2.9274, -1.5762,   -1.9775 ,  -1.3734],
    #    [-2.7709,    7.7694,    2.6515,    1.3540,   -0.3214,    0.3606,    1.0891,  3.1352,    1.8838 ,  -0.0681,],
    #    [-0.4774,    2.6515,    9.9158,   -0.3002,    0.4516,   -2.8206,    2.4770,  0.9234,   -1.5996 ,   1.4107,],
    #    [ 2.2081,    1.3540,   -0.3002,    4.2751,   -0.1384,   -0.3862,    0.1692, -1.8612,   -0.9461 ,  -1.8507,],
    #    [-1.5825,   -0.3214,    0.4516,   -0.1384,    1.4140,   -0.8715,   -0.3199,  0.0210,   -0.2230 ,   0.7343,],
    #    [-0.4233,    0.3606,   -2.8206,   -0.3862,   -0.8715,    2.4307,   -0.0785,  0.4765,    1.3935 ,  -1.8824,],
    #    [-2.9274,    1.0891,    2.4770,    0.1692,   -0.3199,   -0.0785,    5.1824, -0.2687,    0.0974 ,   0.4081,],
    #    [-1.5762,    3.1352,    0.9234,   -1.8612,    0.0210,    0.4765,   -0.2687,  6.6206,   -0.0627 ,   3.5937,],
    #    [-1.9775,    1.8838,   -1.5996,   -0.9461,   -0.2230,    1.3935,    0.0974, -0.0627,    4.4154 ,  -2.6342,],
    #    [-1.3734,   -0.0681,    1.4107,   -1.8507,    0.7343,   -1.8824,    0.4081,  3.5937,   -2.6342 ,   6.2671,],
    # ])
    
    # A = np.loadtxt('A.txt', delimiter=',')

    manifold = BudgetKManifold(n=n, k=k)

    @pymanopt.function.jax(manifold)
    def cost(x):
        return x.T @ A @ x
    # @pymanopt.function.numpy(manifold)
    # def euclidean_gradient(x):
    #     return 2 * A @ x
    x0 = manifold.random_point()
    problem = pymanopt.Problem(manifold, cost, euclidean_gradient=None)
    optimizer = SteepestDescent(verbosity=2)
    result = optimizer.run(problem,  initial_point=x0)
    estimated_min = result.point

    # problem = pymanopt.Problem(manifold, cost,euclidean_gradient=None)
    # optimizer = ConjugateGradient(verbosity=1)

    # profiler = cProfile.Profile()
    # profiler.enable()
    # result = optimizer.run(problem,  initial_point=x0)
    cost = result.cost
    # profiler.disable()
    # profiler.dump_stats('bk_profile.prof')
    # stats= pstats.Stats(profiler)

    print(f"Riemannian cost: {cost}")

    # solve via cvxpy
    x = cvxpy.Variable(n)
    objective = cvxpy.Minimize(cvxpy.quad_form(x, A))
    constraints = [cvxpy.sum(x) == k, x >= 0, x <= 1]
    problem = cvxpy.Problem(objective, constraints)
    problem.solve(solver=cvxpy.ECOS, verbose=True)
    cvxpy_x = x.value
    cvxpy_cost = problem.value

    print(f"CVXPY cost: {cvxpy_cost}")
