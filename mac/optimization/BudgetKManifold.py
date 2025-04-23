import numpy as np
import jax
import jax.numpy as jnp
import functools
from pymanopt.manifolds.manifold import Manifold

class BudgetKManifold(Manifold):
    def __init__(self, n, k=1, run_checks=False, eps_tol=1e-8):
        self.n = n
        self.k = k
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

    @functools.partial(jax.jit, static_argnums=0)
    def inner_product(self, X, eta, zeta):
        denom = (1 - X) * X
        return jnp.sum((eta * zeta) / denom)

    @functools.partial(jax.jit, static_argnums=0)
    def norm(self, X, eta):
        return jnp.sqrt(self.inner_product(X, eta, eta))

    @functools.partial(jax.jit, static_argnums=0)
    def typical_dist(self):
        return np.pi / 4

    @functools.partial(jax.jit, static_argnums=0)
    def euclidean_to_riemannian_gradient(self, X, eta):
        return self.projection(X, eta * X * (1-X))

    @functools.partial(jax.jit, static_argnums=0)
    def projection(self, X, eta):
        alpha = jnp.sum(eta) / jnp.sum(X * (1 - X))
        return eta - alpha * X * (1 - X)

    @functools.partial(jax.jit, static_argnums=0)
    def projection_euclidean(self, X, eta):
        return eta - jnp.mean(eta)

    # @functools.partial(jax.jit, static_argnums=0)
    # def retraction(self, X, U, t=1):
    #     U = self.projection(X, U)
    #     Y = X + t * U
    #     return self.project_to_feasible_set(Y)

    @functools.partial(jax.jit, static_argnums=0)
    def retraction(self, X, U, t=1.0):
        # U = self.projection(X, U)
        max_iters=200
        lower = 0.0
        upper = t
        init_val = (lower, upper, (lower + upper) / 2, X+(lower + upper) / 2 *U, 0)
    
        def cond_fn(state):
            lower, upper, _, f_mid, iter_count = state
            # return jnp.any(jnp.logical_or((iter_count < max_iters),(jnp.abs(upper-lower) > self.eps_tol/10)))
            # return iter_count < max_iters
            convergence_check = jnp.logical_or(jnp.abs(upper-lower) > self.eps_tol,
                jnp.any(jnp.logical_or(jnp.greater(f_mid,(1-self.eps_tol)), jnp.less(f_mid,self.eps_tol)))
            )

            # return jnp.logical_and(convergence_check, iter_count<max_iters)
            return convergence_check
    
        def body_fn(state):
            lower, upper, _, Ynext, iter_count = state
            midpoint = (lower + upper) / 2
            Ynext = X+midpoint * U
        
            # Determine which subinterval to keep
            new_lower, new_upper = jax.lax.cond(
                jnp.any(jnp.logical_or(jnp.max(Ynext)>(1-self.eps_tol), jnp.min(Ynext)<self.eps_tol)),
                lambda: (lower, midpoint),
                lambda: (midpoint, upper),
            )
        
            return (new_lower, new_upper, midpoint, Ynext, iter_count + 1)
        final_state = jax.lax.while_loop(cond_fn, body_fn, init_val)
        lower, upper, solution, f_solution, iters = final_state
        return f_solution
        # return X + lower*U

        

        # # Y=X
        # # for ts in t_span:
        # #     Ynext = X + ts * U
        # #     if max(Ynext)>1 or min(Ynext)<0:
        # #         break
        # #     Y=Ynext
        # return Y
        return self.project_to_feasible_set(X+t*U)

    @functools.partial(jax.jit, static_argnums=0)
    def transport(self, X1, X2, d):
        return self.projection(X2, d)

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
        return self.dykstras_projection_jit(X)

    def dykstras_projection(self, y):
        max_iters = 1000000 
        # tol = self.eps_tol
        tol = 1e-6
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
            # if (np.max(x) < 1 - tol) and (np.min(x) > 0 + tol):
            #     break
            if iter == max_iters - 1:
                print("Reached Max iterations")

        return x

    @functools.partial(jax.jit, static_argnums=0)
    def dykstras_projection_jit(self, y):
        tol = self.eps_tol
        max_iters = 1000
        p = jnp.zeros(self.n)
        q = jnp.zeros(self.n)
        x = y.copy()
        def loop_body(state, iter):
            x, p, q = state

            # Save the previous value of x for convergence check
            x_prev = x.copy()

            # Project onto the hypercube [0, 1]^n (with correction)
            x_tilde = x + q
            x = jnp.clip(x_tilde, 0 + tol, 1 - tol)
            q = x_tilde - x

            # Project onto the k-simplex (with correction)
            x_tilde = x + p
            x = self.project_to_sum_plane(x_tilde)
            p = x_tilde - x

            # Convergence check (this will be handled manually)
            converged = jnp.linalg.norm(x - x_prev) < 1e-12

            return (x, p, q), converged

        def body_fun(state):
            x, p, q, x_prev = state
            # Save the previous value of x for convergence check
            x_prev = x.copy()
            # Project onto the hypercube [0, 1]^n (with correction)
            x_tilde = x + q
            x = jnp.clip(x_tilde, 0 + tol, 1 - tol)
            q = x_tilde - x
            # Project onto the k-simplex (with correction)
            x_tilde = x + p
            x = self.project_to_sum_plane(x_tilde)
            p = x_tilde - x
            # converged = jnp.linalg.norm(x - x_prev) < 1e-18
            return (x, p, q, x_prev)

            # Convergence check (this will be handled manually)
        def cond_fun(state):
            x, p, q, x_prev = state
            # converged = jnp.all((jnp.linalg.norm(x - x_prev) < 1e-8, jnp.all(x < 1-tol), jnp.all(x>0+tol)))
            # converged = jnp.linalg.norm(x - x_prev) < 1e-8 and jnp.max(x) < 1-tol, jnp.min(x)>0+tol
            converged = jnp.linalg.norm(x - x_prev) < 1e-8 
            # return state[-1] 
            return converged

        # Initial state for the scan (x, p, q)
        # init_state = (x, p, q)
        # init_state = (x, p, q, False)
        init_state = (x, p, q, x)
        # Using `jax.lax.scan` to run the loop and accumulate results
        # final_state, converged_values = jax.lax.scan(loop_body, init_state, jnp.arange(max_iters))
        final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)

        # Final state of x, p, q
        x = final_state[0]
        return x

    @functools.partial(jax.jit, static_argnums=0)
    def project_to_sum_plane(self, pt):
        dist_to_plane = self.k - jnp.sum(pt)
        return pt + dist_to_plane / self.n
    '''

    def project_pt_on_plane_also_onto_cube(self, pt):
        clipped_pt = np.clip(pt, 0 + self.eps_tol, 1 - self.eps_tol)
        diff = clipped_pt - pt
        active_dirs = np.nonzero(diff)[0]
        
        if active_dirs.size == 0:
            return clipped_pt
        
        diff_tangent = self.projection(pt, diff)
        scale = np.max(np.abs(diff[active_dirs] / diff_tangent[active_dirs]))
        return pt + scale * diff_tangent

    def project_pt_on_plane_along_cent(self, pt):
        clipped_pt = np.clip(pt, 0 + self.eps_tol, 1 - self.eps_tol)
        diff = clipped_pt - pt
        active_dirs = np.nonzero(diff)[0]
        
        if active_dirs.size == 0:
            return clipped_pt
        
        centroid = np.ones(self.n) * self.k / self.n
        vec = centroid - pt
        vec /= np.sum(np.abs(vec))
        scale = np.max(np.abs(diff[active_dirs] / vec[active_dirs]))
        return pt + scale * vec

    def project_onto_simplex_hypercube_intersection(self, y):
        max_iters = 1000
        tol = self.eps_tol

        x = np.clip(y, 0 + tol, 1 - tol)
        current_sum = np.sum(x)

        if abs(current_sum - self.k) < tol:
            return x

        low = np.min(y) - 1
        high = np.max(y) + 1

        for iter in range(max_iters):
            theta = (low + high) / 2
            x_proj = np.clip(y - theta, 0 + tol, 1 - tol)
            sum_proj = np.sum(x_proj)
            
            if abs(sum_proj - self.k) < tol:
                break
            elif sum_proj < self.k:
                high = theta
            else:
                low = theta

        return x_proj
    '''

if __name__=="__main__":
    import numpy as np
    import pymanopt
    from pymanopt.optimizers import SteepestDescent, ConjugateGradient
    from pymanopt.optimizers.line_search import BackTrackingLineSearcher
    import cProfile
    import pstats
    k = 625
    n = 1000

    np.random.seed(2002)
    # A = np.random.randn(n, n)
    # A = A @ A.T

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
    
    A = np.loadtxt('A.txt', delimiter=',')

    manifold = BudgetKManifold(n=n, k=k)

    @pymanopt.function.jax(manifold)
    def cost(x):
        return x.T @ A @ x
    # @pymanopt.function.numpy(manifold)
    # def euclidean_gradient(x):
    #     return 2 * A @ x
    x0 = manifold.random_point()

    problem = pymanopt.Problem(manifold, cost, euclidean_gradient=None)
    optimizer = ConjugateGradient(verbosity=1)
    result = optimizer.run(problem,  initial_point=x0)
    estimated_min = result.point

    problem = pymanopt.Problem(manifold, cost,euclidean_gradient=None)
    optimizer = ConjugateGradient(verbosity=1)

    profiler = cProfile.Profile()
    profiler.enable()
    result = optimizer.run(problem,  initial_point=x0)
    cost = result.cost
    profiler.disable()
    profiler.dump_stats('bk_profile.prof')
    stats= pstats.Stats(profiler)

    print(cost)
