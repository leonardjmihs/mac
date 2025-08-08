import time
from copy import deepcopy

import numpy as np

from pymanopt.optimizers.optimizer import Optimizer, OptimizerResult
from pymanopt.tools import printer
from mac.optimization.line_search import *

class NesterovDescent(Optimizer):
    """Riemannian steepest descent algorithm.

    Perform optimization using gradient descent with line search.
    This method first computes the gradient of the objective, and then
    optimizes by moving in the direction of steepest descent (which is the
    opposite direction to the gradient).

    Args:
        line_searcher: The line search method.
    """

    def __init__(self, momentum_searcher=None, step_size_searcher=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # if step_size is None:
        #     self._step_size = lambda x, k: naive_stepsize(k)
        # else:
        #     self._step_size = step_size
        if step_size_searcher is None:
            self._step_size_searcher=ConstantLineSearcher(0.5)
        else:
            self._step_size_searcher=step_size_searcher
        
        if momentum_searcher is None:
            self._momentum_searcher=ConstantLineSearcher(0.5)
        else:
            self._momentum_searcher=momentum_searcher

    def reset_log(self):
        self._log={'iterations':{"iteration":[],"time":[], "point":[], "cost":[],"gradient_norm":[]}}
    
    def reset_step_size_searcher(self):
        self._step_size_searcher.reset()

    # Function to solve optimisation problem using steepest descent.
    def run(
        self, manifold, problem, *, initial_point=None    
        ) -> OptimizerResult:
        """Run steepest descent algorithm.

        Args:
            problem: Pymanopt problem class instance exposing the cost function
                and the manifold to optimize over.
                The class must either
            initial_point: Initial point on the manifold.
                If no value is provided then a starting point will be randomly
                generated.
            reuse_line_searcher: Whether to reuse the previous line searcher.
                Allows to use information from a previous call to
                :meth:`solve`.

        Returns:
            Local minimum of the cost function, or the most recent iterate if
            algorithm terminated before convergence.
        """
        # manifold = problem.manifold
        # objective = problem.cost
        # gradient = problem.riemannian_gradient

        step_size_searcher = self._step_size_searcher
        momentum_searcher = self._momentum_searcher

        # If no starting point is specified, generate one at random.
        if initial_point is None:
            x = manifold.random_point()
        else:
            x = initial_point

        if self._verbosity >= 1:
            print("Optimizing...")
        if self._verbosity >= 2:
            iteration_format_length = int(np.log10(self._max_iterations)) + 1
            column_printer = printer.ColumnPrinter(
                columns=[
                    ("Iteration", f"{iteration_format_length}d"),
                    ("Cost", "+.16e"),
                    ("Gradient norm", ".8e"),
                    ("Step Size", ".8e"),
                ]
            )
        else:
            column_printer = printer.VoidPrinter()

        column_printer.print_header()

        self._initialize_log(
            optimizer_parameters={"step_size": step_size_searcher}
        )

        # Initialize iteration counter and timer
        iteration = 0
        start_time = time.time()
        # velocity = manifold.zero_vector(x)
        v = x
        A = 0
        cost, egrad = problem(x)
        grad = manifold.euclidean_to_riemannian_gradient(x, egrad)
        gradient_norm = manifold.norm(x, grad)

        while True:
            iteration += 1

            # Calculate new cost, grad and gradient_norm
            # cost = objective(x)
            # grad = gradient(x)
            # momentum, _ = momentum_searcher.search(
            #     problem, manifold, x, v, cost, -(gradient_norm**2) # Currently only fixed parameter momenutm is implemented
            # )
            momentum = iteration/(iteration+2)
            # momentum = 0.5 
            # y  = manifold.retraction(v, momentum*manifold.log(x,v))
            y  = manifold.retraction(v, momentum*manifold.log(v,x))

            next_cost, egrad = problem(y)
            grad = manifold.euclidean_to_riemannian_gradient(x, egrad)
            gradient_norm = manifold.norm(x, grad)

            relative_cost_reduction = abs((next_cost-cost) / cost )

            # velocity = manifold.retraction(x, self.momentum*(x-x_prev))

            # Descent direction is minus the gradient
            desc_dir = -grad


            step_size, xNext = step_size_searcher.search(
                problem, manifold, y, desc_dir, cost, -(gradient_norm**2), iter=iteration
            )
            a = np.max(np.roots(np.array([1, -step_size, -step_size*A])))
            A = A + a
            column_printer.print_row([iteration, cost, gradient_norm, momentum])
            self._add_log_entry(
                iteration=iteration,
                point=x,
                cost=cost,
                gradient_norm=gradient_norm,
                step_size=momentum
            )

            v = manifold.retraction(v, a*manifold.transport(y, v,desc_dir))
            x = xNext 

            stopping_criterion = self._check_stopping_criterion(
                start_time=start_time,
                step_size=momentum,
                gradient_norm=gradient_norm,
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
            point=x,
            cost=cost,
            iterations=iteration,
            stopping_criterion=stopping_criterion,
            cost_evaluations=iteration,
            step_size=step_size,
            gradient_norm=gradient_norm,
        )