import numpy as np
import time
from pymanopt.optimizers.optimizer import Optimizer, OptimizerResult
from pymanopt.tools import printer
from mac.optimization.line_search import *


def matrixlincomb(a1, d1, a2=None, d2=None):
    if a2 is None:
        return a1 * d1
    else:
        return a1*d1 + a2*d2

class rlbfgs(Optimizer):
    def __init__(self, memory=30, step_size_searcher=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if step_size_searcher is None:
            self._step_size_searcher = ConstantAlphaLineSearcher()
        else:
            self._step_size_searcher = step_size_searcher
        self.strict_inc_func = lambda t: 1e-8 * t
        self.ls_initial_scale = lambda gradnorm: 1 / gradnorm
        self.memory = memory
        self._log={'iterations':{"iteration":[],"time":[], "point":[], "cost":[],"gradient_norm":[]}}

    def reset_log(self):
        self._log={'iterations':{"iteration":[],"time":[], "point":[], "cost":[],"gradient_norm":[]}}

    def reset_step_size_searcher(self):
        self._step_size_searcher.reset()

    def run(
        self, manifold, problem, *, initial_point=None,    
        ):
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
        
        xCurCost, exCurGradient = problem(xCur)
        xCurGradient = manifold.euclidean_to_riemannian_gradient(xCur, exCurGradient)
        xCurGradNorm = manifold.norm(xCur, xCurGradient)
        scaleFactor = self.ls_initial_scale(xCurGradNorm)
        lsstats = []

        # Iteration statistics
        if self._verbosity >= 1:
            print("Optimizing...")
        if self._verbosity >= 2:
            iteration_format_length = int(np.log10(self._max_iterations)) + 1
            column_printer = printer.ColumnPrinter(
                columns=[
                    ("Iteration", f"{iteration_format_length}d"),
                    ("Cost", "+.16e"),
                    ("Gradient norm", ".8e"),
                    ("Relative Cost reduction", ".8e"),
                    ("Alpha", ".8e"),
                    ("Step Size", ".8e")
                ]
            )
        else:
            column_printer = printer.VoidPrinter()
        iteration = 1
        start_time = time.time()
        column_printer.print_header()
        column_printer.print_row([0, xCurCost, xCurGradNorm, 0.0, 0.0, 0.0])

        lincomb = matrixlincomb
        if hasattr(manifold, 'lincomb'):
            lincomb = manifold.lincomb

        while True:
            # Compute BFGS direction

            self._add_log_entry(iteration=iteration,point= xCur, cost=xCurCost, gradient_norm=xCurGradNorm)
            p = self.get_direction(manifold, xCur, xCurGradient, sHistory, yHistory, rhoHistory, scaleFactor, min(k, self.memory))
            p = manifold.projection(xCur, p)
            # Execute line search (assuming linesearch_hint is implemented)
            # stepsize, xNext = self._step_size_searcher.search(xCur, iteration)
            stepsize, xNext = self._step_size_searcher.search(
                problem, manifold, xCur, p, xCurCost, -(xCurGradNorm**2), iter=iteration
            )

            # Record the BFGS step-multiplier
            alpha = stepsize / manifold.norm(xCur, p)
            step = lincomb(alpha, p)

            # Query cost and gradient at the candidate new point
            xNextCost, exNextGrad = problem(np.array(xNext))
            xNextGrad = manifold.euclidean_to_riemannian_gradient(xNext, exNextGrad)
        
            # Compute sk and yk
            sk = manifold.transport(xCur, xNext, step)
            yk = lincomb(1, xNextGrad, -1, manifold.transport(xCur, xNext, xCurGradient))

            # Normalize sk and yk
            norm_sk = manifold.norm(xNext, sk)
            sk = lincomb(1 / norm_sk, sk)
            yk = manifold.projection(xNext, lincomb(1 / norm_sk, yk))
        
            inner_sk_yk = manifold.inner_product(xNext, sk, yk)
            inner_sk_sk = manifold.norm(xNext, sk) ** 2

            # Cautious update check (accept/reject step)
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
            relative_cost_reduction = abs((xNextCost-xCurCost) / xCurCost)
            # if not manifold.check_on_manifold(xNext):
            #     breakpoint()
            xCur = xNext
            xCurGradient = xNextGrad
            xCurGradNorm = manifold.norm(xNext, xNextGrad)
            xCurCost = xNextCost

            column_printer.print_row([iteration, xCurCost, xCurGradNorm, relative_cost_reduction, alpha, stepsize])


            # Increment iteration counter
            iteration += 1

            stopping_criterion = self._check_stopping_criterion(
                start_time=start_time,
                step_size=stepsize,
                gradient_norm=xCurGradNorm,
                iteration=iteration,
                relative_cost_reduction=relative_cost_reduction
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

        lincomb = matrixlincomb
        if hasattr(M, 'lincomb'):
            lincomb = M.lincomb
        q = xCurGradient
    
        # Create an array to store inner products
        inner_s_q = [0] * k
    
        # First loop for the backward pass (equivalent to MATLAB's for i = k : -1 : 1)
        for i in range(k - 1, -1, -1):
            inner_s_q[i] = rhoHistory[i] * M.inner_product(xCur, sHistory[i], q)
            q = lincomb(1, q, -inner_s_q[i], yHistory[i])
    
        # Now apply the scaling factor and linear combination
        r = lincomb( scaleFactor, q)
    
        # Second loop for the forward pass (equivalent to MATLAB's for i = 1 : k)
        for i in range(k):
            omega = rhoHistory[i] * M.inner_product(xCur, yHistory[i], r)
            r = lincomb(1, r, inner_s_q[i] - omega, sHistory[i])
    
        # Final direction with negative sign
        dir = lincomb(-1, r)
    
        return dir

def test_main():
    import numpy as np
    import cProfile
    import pstats
    from pymanopt.manifolds import Sphere
    k = 5
    n = 10

    A = np.array([8.03433465544415,-0.872215350953547,0.811414302125796,2.46131457816524,-3.30856248976592,-1.66006998830024,3.16311526341609,4.24009304011803,5.75703652412681,-0.236822446367289,
                  -0.872215350953547,4.88229000024012,2.05581186707058,1.69399056672827,0.364940598432017,0.750057974817370,-0.651486853306673,-4.25938430828979,-2.57795305624483,2.54799335937894,
                  0.811414302125796,2.05581186707058,8.30228157300619,0.582821530955579,-0.934606009636986,-0.215914424989828,-2.83358776891198,0.420436765852012,-1.28757850178764,1.10060506455263,
                  2.46131457816524,1.69399056672827,0.582821530955579,7.16492630055535,-1.87855769677988,-1.25668522436733,1.33420748750929,-2.03575375726126,5.74863593261962,-0.0668498412342219,
                  -3.30856248976592,0.364940598432017,-0.934606009636986,-1.87855769677988,4.86778345535167,1.53515467946349,-1.44158087772233,-1.42311629596462,-3.09037132578349,-0.964423035427173,
                  -1.66006998830024,0.750057974817370,-0.215914424989828,-1.25668522436733,1.53515467946349,3.20463397779291,-0.410068722105824,0.489076959510760,-3.80052555036089,-0.0167350158297679,
                  3.16311526341609,-0.651486853306673,-2.83358776891198,1.33420748750929,-1.44158087772233,-0.410068722105824,4.05792114641302,2.15064778911319,3.80756035551578,-0.397986887522710,
                  4.24009304011803,-4.25938430828979,0.420436765852012,-2.03575375726126,-1.42311629596462,0.489076959510760,2.15064778911319,9.69906105933449,4.18349002375522,-2.33826871039923,
                  5.75703652412681,-2.57795305624483,-1.28757850178764,5.74863593261962,-3.09037132578349,-3.80052555036089,3.80756035551578,4.18349002375522,12.9247204012558,-3.69059594944703,
                  -0.236822446367289,2.54799335937894,1.10060506455263,-0.0668498412342219,-0.964423035427173,-0.0167350158297679,-0.397986887522710,-2.33826871039923,-3.69059594944703,4.71109714870801]).reshape(10,10)

    manifold = Sphere(n)

    def cost(x):
        return x.T @ A @ x, 2 * A @ x

    x0 = np.array([-0.2826,
                        0.0319,
                    -0.1733,
                        0.0966,
                    -0.1910,
                        0.1559,
                        0.2353,
                        0.5448,
                    -0.0618,
                    -0.6805]).T

    optimizer = rlbfgs(verbosity=2, max_iterations=20)
    result = optimizer.run(manifold, cost,  initial_point=x0)
    breakpoint()


    profiler = cProfile.Profile()
    profiler.enable()
    result = optimizer.run(cost,  initial_point=x0)
    cost = result.cost
    profiler.disable()
    profiler.dump_stats('bk_profile.prof')
    stats= pstats.Stats(profiler)

    print(cost)

if __name__=="__main__":
    test_main()