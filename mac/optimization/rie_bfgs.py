import numpy as np
import time
from pymanopt.optimizers.optimizer import Optimizer, OptimizerResult
from pymanopt.tools import printer

def naive_stepsize(k):
    return 2.0 / (k + 2.0)
    # return 0.1

class rlbfgs(Optimizer):

    def __init__(self, memory=30, step_size=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if step_size is None:
            self._step_size = lambda x, k: naive_stepsize(k)
        else:
            self._step_size = step_size
        self.strict_inc_func = lambda t: 1e-4 * t
        self.ls_initial_scale = lambda gradnorm: 1 / gradnorm
        self.memory = memory
        self._log={'iterations':{"iteration":[],"time":[], "point":[], "cost":[],"gradient_norm":[]}}
    def reset_log(self):
        self._log={'iterations':{"iteration":[],"time":[], "point":[], "cost":[],"gradient_norm":[]}}
    def run(
        self, manifold, problem, *, initial_point=None,    
        ):
            #  manifold, problem, x0=None, options=None):
        # Random initialization if x0 is not provided
        x0 = initial_point
        if x0 is None:
            xCur = manifold.random_point()
        else:
            xCur = x0

        # Initialize iteration variables
        k = 0
        options = {'memory':self.memory}
        sHistory = [None] * options['memory']
        yHistory = [None] * options['memory']
        rhoHistory = [None] * options['memory']
        alpha = 1
        stepsize = 1
        accepted = True

        # Get initial cost and gradient
        
        xCurCost, xCurGradient = problem(xCur)
        xCurGradNorm = manifold.norm(xCur, xCurGradient)
        scaleFactor = self.ls_initial_scale(xCurGradNorm)
        lsstats = []

        # Iteration statistics
        # info = [stats]
    
        if self._verbosity >= 1:
            print("Optimizing...")
        if self._verbosity >= 2:
            iteration_format_length = int(np.log10(self._max_iterations)) + 1
            column_printer = printer.ColumnPrinter(
                columns=[
                    ("Iteration", f"{iteration_format_length}d"),
                    ("Cost", "+.16e"),
                    ("Gradient norm", ".8e"),
                ]
            )
        else:
            column_printer = printer.VoidPrinter()
        iteration = 0
        start_time = time.time()

        column_printer.print_row([iteration, xCurCost, xCurGradNorm])

        while True:
            # Compute BFGS direction

            # xCurCost, xCurGradient = problem(xCur)
            # xCurGradNorm = manifold.norm(xCur, xCurGradient)

            self._add_log_entry(iteration=iteration,point= xCur, cost=xCurCost, gradient_norm=xCurGradNorm)
            p = self.get_direction(manifold, xCur, xCurGradient, sHistory, yHistory, rhoHistory, scaleFactor, min(k, self.memory))
            p = manifold.projection(xCur, p)
            # Execute line search (assuming linesearch_hint is implemented)
            stepsize = self._step_size(xCur, iteration)
            xNext = manifold.retraction(xCur, alpha*p)
            # print(sum(xNext))
            # print(np.min(xNext))
            # print(np.max(xNext))
            # breakpoint()
            # stepsize, xNext = linesearch_hint(problem, xCur, p, xCurCost, M.inner(xCur, xCurGradient, p), options, storedb, key)

            # Record the BFGS step-multiplier

            alpha = stepsize / manifold.norm(xCur, p)
            step = manifold.lincomb(alpha, p)

            # Query cost and gradient at the candidate new point
            # xNextCost, xNextGrad = problem(np.array(xCur))
            xNextCost, xNextGrad = problem(np.array(xNext))
            xNextGrad = manifold.euclidean_to_riemannian_gradient(xCur, xNextGrad)
        
            # Compute sk and yk
            sk = manifold.transport(xCur, xNext, step)
            yk = manifold.lincomb(1, xNextGrad, -1, manifold.transport(xCur, xNext, xCurGradient))

            # Normalize sk and yk
            norm_sk = manifold.norm(xNext, sk)
            sk = manifold.lincomb(1 / norm_sk, sk)
            yk = manifold.projection(xNext, manifold.lincomb(1 / norm_sk, yk))
        
            inner_sk_yk = manifold.inner_product(xNext, sk, yk)
            inner_sk_sk = manifold.norm(xNext, sk) ** 2

            # Cautious update check (accept/reject step)
            # cap = options['strict_inc_func'](xCurGradNorm)
            cap = self.strict_inc_func(xCurGradNorm)
            if inner_sk_sk != 0 and (inner_sk_yk / inner_sk_sk) >= cap:
                accepted = True
                rhok = 1 / inner_sk_yk
                scaleFactor = inner_sk_yk / manifold.norm(xNext, yk) ** 2

                # Store vectors sk, yk, and rhok
                if k >= options['memory']:
                    # Shift old entries in history
                    for i in range(1, options['memory']):
                        sHistory[i] = manifold.transport(xCur, xNext, sHistory[i])
                        yHistory[i] = manifold.transport(xCur, xNext, yHistory[i])
                    if options['memory'] > 1:
                        sHistory = [sHistory[-1]] + sHistory[:-1]
                        yHistory = [yHistory[-1]] + yHistory[:-1]
                        rhoHistory = [rhoHistory[-1]] + rhoHistory[:-1]
                else:
                    for i in range(k):
                        sHistory[i] = manifold.transport(xCur, xNext, sHistory[i])
                        yHistory[i] = manifold.transport(xCur, xNext, yHistory[i])
                    sHistory[k] = sk
                    yHistory[k] = yk
                    rhoHistory[k] = rhok
            
                k += 1
            else:
                accepted = False
                for i in range(min(k, options['memory'])):
                    sHistory[i] = manifold.transport(xCur, xNext, sHistory[i])
                    yHistory[i] = manifold.transport(xCur, xNext, yHistory[i])

            # Update variables to new iterate
            # storedb.removefirstifdifferent(key, newkey)
            # column_printer.print_row([iteration+1, xNextCost, manifold.norm(xNext, xNext)])
            column_printer.print_row([iteration, xCurCost, xCurGradNorm])
            if not manifold.check_on_manifold(xNext):
                breakpoint()
            effective_step = np.linalg.norm(xCur-xNext)
            xCur = xNext
            # key = newkey
            xCurGradient = xNextGrad
            xCurGradNorm = manifold.norm(xNext, xNextGrad)
            xCurCost = xNextCost
            

            # Increment iteration counter
            iteration += 1

            # Purge storedb to limit memory usage
            # info.append(stats)

            stopping_criterion = self._check_stopping_criterion(
                start_time=start_time,
                step_size=effective_step,
                gradient_norm=xCurGradNorm,
                iteration=iteration,
            )
            if stopping_criterion:
                if self._verbosity >= 1:
                    print(stopping_criterion)
                    print("")
                break

        return self._return_result(
            start_time=start_time,
            point=xCur,
            cost=xCurCost,
            iterations=iteration,
            stopping_criterion=stopping_criterion,
            cost_evaluations=iteration,
            step_size=stepsize,
            gradient_norm=xCurGradNorm,
        )

    def get_direction(self,M, xCur, xCurGradient, sHistory, yHistory, rhoHistory, scaleFactor, k):
        # Initialize q as the current gradient
        q = xCurGradient
    
        # Create an array to store inner products
        inner_s_q = [0] * k
    
        # First loop for the backward pass (equivalent to MATLAB's for i = k : -1 : 1)
        for i in range(k - 1, -1, -1):
            inner_s_q[i] = rhoHistory[i] * M.inner_product(xCur, sHistory[i], q)
            q = M.lincomb(1, q, -inner_s_q[i], yHistory[i])
    
        # Now apply the scaling factor and linear combination
        r = M.lincomb( scaleFactor, q)
    
        # Second loop for the forward pass (equivalent to MATLAB's for i = 1 : k)
        for i in range(k):
            omega = rhoHistory[i] * M.inner_product(xCur, yHistory[i], r)
            r = M.lincomb(1, r, inner_s_q[i] - omega, sHistory[i])
    
        # Final direction with negative sign
        dir = M.lincomb(-1, r)
    
        return dir