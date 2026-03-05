import numpy as np
import networkx as nx
import networkx.linalg as la

from dataclasses import dataclass
from typing import Optional
from timeit import default_timer as timer

from mac.utils.graphs import *
from mac.utils.rounding import *

import mac.utils.fiedler as fiedler
import mac.optimization.frankwolfe as fw
import mac.optimization.constraints as constraints
from mac.optimization.line_search import *
from mac.optimization.BudgetKManifold import BudgetKManifold
from mac.optimization.rie_gd import SteepestDescent
from mac.optimization.rie_bfgs import rlbfgs
from mac.optimization.rie_nm import NesterovDescent

import pymanopt

class MAC:
    @dataclass
    class Cache:
        """Problem data cache"""
        Q: Optional[np.ndarray] = None

    def __init__(self, fixed_edges, candidate_edges, num_nodes,
                fiedler_method='tracemin_lu', fiedler_tol=1e-8,
                min_selection_weight_tol=1e-10, memory=10,
                step_size_searcher=None,
                rie_alg="rbfgs"):
        """Parameters
        ----------
        fixed_edges : list of Edge
            List of edges that are fixed in the graph.
        candidate_edges : list of Edge
            List of edges that are candidates for addition to the graph.
        num_nodes : int
            Number of nodes in the graph.
        fiedler_method : str, optional
            Method to use for computing the Fiedler vector. Options are
            'tracemin_lu', 'tracemin_cholesky'. Default is 'tracemin_lu'. Using the
            'tracemin_cholesky' method is faster but requires SuiteSparse to be
            installed.
        fiedler_tol : float, optional
            Tolerance for computing the Fiedler vector and corresponding eigenvalue.
        min_edge_selection_tol : float, optional
            Tolerance for the minimum edge selection weight. Default is 1e-10.
        """
        # Check that we at least *could* have a spanning tree in the set
        # {fixed_edges U candidate_edges} This does not guarantee that a
        # spanning tree exists, but it's a good basic test.
        num_edges = len(fixed_edges) + len(candidate_edges)
        assert (num_nodes - 1) <= num_edges

        # We also check that there aren't "too many" edges. The number of edges
        # in the complete graph K(n) is equal to n * (n - 1) / 2, so we cannot
        # possibly have more edges than this.
        assert num_edges <= 0.5 * num_nodes * (num_nodes - 1)

        # Pre-compute the Laplacian for the subgraph comprised of the "fixed edges".
        self.L_fixed = weight_graph_lap_from_edge_list(fixed_edges, num_nodes)
        self.num_nodes = num_nodes

        self.weights = []
        self.edge_list = []
        for edge in candidate_edges:
            self.weights.append(edge.weight)
            self.edge_list.append((edge.i, edge.j))

        self.weights = np.array(self.weights)
        self.edge_list = np.array(self.edge_list)

        # Configuration for Fiedler vector computation
        self.fiedler_method = fiedler_method
        self.fiedler_tol = fiedler_tol

        # Truncate edges with selection weights below this threshold
        self.min_selection_weight_tol = min_selection_weight_tol
        # self.optimizer = pymanopt.optimizers.ConjugateGradient(verbosity=2, max_iterations=100, min_step_size=1e-3)
        # self.optimizer = pymanopt.optimizers.SteepestDescent(verbosity=2, max_iterations=100, min_step_size=1e-3)

        if step_size_searcher is None:
            # step_size_searcher = ConstantAlphaLineSearcher(alpha=1.0)
            # step_size_searcher = ConstantStepSizeLineSearcher(step_size=1.0)
            step_size_searcher = AdaptiveLineSearcher()

        if rie_alg == "rbfgs":
            self.optimizer = rlbfgs(min_step_size=1e-8, log_verbosity=2, memory=memory, step_size_searcher=step_size_searcher,  min_relative_cost_reduction=1e-8)
        elif rie_alg == "rgd":
            self.optimizer = SteepestDescent(log_verbosity=2, step_size_searcher=step_size_searcher, min_step_size=1e-8, min_relative_cost_reduction=1e-8)
        else:
            raise ValueError(f"Unknown Riemannian optimization algorithm: {rie_alg}. Choose 'rbfgs' or 'rgd'.")

        # self.optimizer = rlbfgs(verbosity=2, max_iterations=100, min_step_size=1e-5, log_verbosity=2, memory=30, step_size_searcher=ConstantLineSearcher(),  min_relative_cost_reduction=5e-4)
        # self.optimizer = rlbfgs(verbosity=2, max_iterations=100, min_step_size=1e-10, log_verbosity=2, memory=10, step_size_searcher=NaieveLineSearcher(),  min_relative_cost_reduction=1e-10)
        # self.optimizer = NesterovDescent(verbosity=2, max_iterations=20, min_step_size=1e-6, log_verbosity=2)

        # self.prob_run_info = {'runtimes':[], 'calls': 0, 'num_idx':[], 'lap_rt':[], 'weight_graph_count':0, 'wgle_rt':[]}
        # self.prob_rie_run_info = {'runtimes':[], 'calls': 0, 'num_idx':[], 'lap_rt':[],'weight_graph_count':0, 'wgle_rt':[]}
        self.prob_run_info = {'runtimes':[], 'calls': 0, 'num_idx':[], 'lap_rt':[],'weight_graph_count':0, 'wgle_rt':[], 'fiedler_iters':[], 'fiedler_runtimes':[], 'ratio':[]}
        self.prob_rie_run_info = {'runtimes':[], 'calls': 0, 'num_idx':[], 'lap_rt':[],'weight_graph_count':0, 'wgle_rt':[], 'fiedler_iters':[], 'fiedler_runtimes':[], 'ratio':[]}

        self.mac_log = {"iterations": {"iteration": [], "time": [], "point": [], "cost": [], "gradient_norm": []}}

    def reset_log(self):
        self.optimizer.reset_log()
        self.mac_log = {"iterations": {"iteration": [], "time": [], "point": [], "cost": [], "gradient_norm": []}}
        # self.prob_run_info = {'runtimes':[], 'calls': 0, 'num_idx':[], 'lap_rt':[], 'weight_graph_count':0, 'wgle_rt':[]}
        # self.prob_rie_run_info = {'runtimes':[], 'calls': 0, 'num_idx':[], 'lap_rt':[],'weight_graph_count':0, 'wgle_rt':[]}

        self.prob_run_info = {'runtimes':[], 'calls': 0, 'num_idx':[], 'lap_rt':[],'weight_graph_count':0, 'wgle_rt':[], 'fiedler_iters':[], 'fiedler_runtimes':[], 'ratio':[]}
        self.prob_rie_run_info = {'runtimes':[], 'calls': 0, 'num_idx':[], 'lap_rt':[],'weight_graph_count':0, 'wgle_rt':[], 'fiedler_iters':[], 'fiedler_runtimes':[], 'ratio':[]}

    def laplacian(self, x, solver='fw'):
        """Construct the combined Laplacian (fixed edges plus candidate edges weighted by x).
        x: An element of [0,1]^m; this is the edge selection to use

        The tolerance parameter `min_selection_weight_tol` is used to prune out
        edges that are numerically zero. This improves speed in situations
        where edges are not *exactly* zero, but close enough that they have
        almost no influence on the graph.

        returns the matrix L(x)
        """
        start = timer()
        idx = np.where(x > self.min_selection_weight_tol)
        prod = x[idx]*self.weights[idx]
        start2 = timer()
        L_candidate = weight_graph_lap_from_edges(self.edge_list[idx], prod, self.num_nodes)
        end2 = timer()
        L_x = self.L_fixed + L_candidate
        end = timer()
        if solver == 'fw':
            self.prob_run_info['lap_rt'].append(end - start)
            self.prob_run_info['num_idx'].append(len(idx[0]))
            self.prob_run_info['wgle_rt'].append(end2-start2)
        else:
            self.prob_rie_run_info['lap_rt'].append(end - start)
            self.prob_rie_run_info['num_idx'].append(len(idx[0]))
            self.prob_rie_run_info['wgle_rt'].append(end2-start2)
        return L_x

    def evaluate_objective(self, x):
        """
        Compute lambda_2(L(x)) where L(x) is the Laplacian with edge i weighted
        by x_i*weight_i and lambda_2 is the second smallest eigenvalue (this is the
        algebraic connectivity).

        x: Weights for each candidate edge (does not include fixed edges)

        returns F(x) = lambda_2(L(x)).
        """
        return fiedler.find_fiedler_pair(L=self.laplacian(x),
                                         method=self.fiedler_method, tol=self.fiedler_tol)[0]

    def problem(self, x, cache=None):
        """Compute the algebraic connectivity of L(x) and a (super)gradient of the
        algebraic connectivity with respect to x.

        x: Weights for each candidate edge (does not include fixed edges)
        cache: Mutable `Cache` object. If a `Cache` object is provided in the `cache` field, it will be used
        and updated, but not explicitly returned. Rather, it will be updated directly.

        returns x, grad F(x).
        """
        start = timer()
        Q = None if cache is None else cache.Q
        # f, fiedler_vec, Qnew = fiedler.find_fiedler_pair(L=self.laplacian(x,solver='fw'), X=Q, method='tracemin_cholesky')
        (f, fiedler_vec, Qnew), (average_time, iters,  ratio) = fiedler.find_fiedler_pair(L=self.laplacian(x, solver='fw'), X=Q, method='tracemin_cholesky')
        end = timer()
        # f, fiedler_vec, Qnew = fiedler.find_fiedler_pair(L=self.laplacian(x, solver='rie'), X=Q, method='tracemin_pcg')
        gradf = np.zeros(len(self.weights))
        for k in range(len(self.weights)):
            edge = self.edge_list[k] # get edge (i,j)
            v_i = fiedler_vec[edge[0]]
            v_j = fiedler_vec[edge[1]]
            weight_k = self.weights[k]
            kdelta = weight_k * (v_i - v_j)
            gradf[k] = kdelta * (v_i - v_j)

        if cache is not None:
            cache.Q = Q

        # end = timer()
        self.prob_run_info['calls'] += 1
        self.prob_run_info['runtimes'].append(end - start)
        self.prob_run_info['fiedler_iters'].append(iters)
        self.prob_run_info['fiedler_runtimes'].append(average_time)
        self.prob_run_info['ratio'].append(ratio)
        return f, gradf

    def problem_rie(self, x, cache=None):
        """Compute the algebraic connectivity of L(x) and a (super)gradient of the
        algebraic connectivity with respect to x.

        x: Weights for each candidate edge (does not include fixed edges)
        cache: Mutable `Cache` object. If a `Cache` object is provided in the `cache` field, it will be used
        and updated, but not explicitly returned. Rather, it will be updated directly.

        returns x, grad F(x).
        """
        start = timer()
        Q = None if cache is None else cache.Q
        (f, fiedler_vec, Qnew), (average_time, iters, ratio) = fiedler.find_fiedler_pair(L=self.laplacian(x, solver='rie'), X=Q, method='tracemin_cholesky')
        end = timer()
        # f, fiedler_vec, Qnew = fiedler.find_fiedler_pair(L=self.laplacian(x, solver='rie'), X=Q, method='tracemin_pcg')

        gradf = np.zeros(len(self.weights))
        for k in range(len(self.weights)):
            edge = self.edge_list[k] # get edge (i,j)
            v_i = fiedler_vec[edge[0]]
            v_j = fiedler_vec[edge[1]]
            weight_k = self.weights[k]
            kdelta = weight_k * (v_i - v_j)
            gradf[k] = kdelta * (v_i - v_j)

        if cache is not None:
            cache.Q = Q

        # end = timer()
        self.prob_rie_run_info['calls'] += 1
        self.prob_rie_run_info['runtimes'].append(end - start)
        self.prob_rie_run_info['fiedler_iters'].append(iters)
        self.prob_rie_run_info['fiedler_runtimes'].append(average_time)
        self.prob_rie_run_info['ratio'].append(ratio)
        return -f, -gradf
        # return f, gradf

    def solve(self, k, x_init=None, rounding="nearest", fallback=False,
              max_iters=5, relative_duality_gap_tol=1e-4, min_relative_cost_reduction=5e-4,
              grad_norm_tol=1e-8, random_rounding_max_iters=1,
              verbose=False, return_rounding_time=False, use_cache=False):
        """Use the Frank-Wolfe method to solve the subset selection problem,.

        Parameters
        ----------
        k : int
            Number of edges to select.
        x_init : optional, array-like
            Initial weights for the candidate edges, must satisfy 0 <= w_i <= 1, |w| <= k. This
            is the starting point for the Frank-Wolfe algorithm. TODO(kevin): make optional
        rounding : str, optional
            Rounding method to use. Options are "nearest" (default) and "madow"
            (a random rounding procedure).
        fallback : bool, optional
            If True, fall back to the initialization if the rounded solution is worse.
        max_iters: int, optional
            Maximum number of iterations for the Frank-Wolfe algorithm.
        relative_duality_gap_tol: float, optional
            Tolerance for the relative duality gap, expressed as a fraction of
            the function value. That is, if (upper - f)/f <
            relative_duality_gap_tol, where "upper" is an upper bound on the
            optimal value of 'f', the algorithm terminates.
        grad_norm_tol: float, optional
            Tolerance for the norm of the gradient. If the norm of the gradient
            is less than this value, then the algorithm terminates.
        random_rounding_max_iters: int, optional
            Maximum number of iterations for the random rounding procedure.
            This is only used if rounding="madow". If this is larger than 1,
            then we will randomly round multiple times and return the best
            solution (in terms of algebraic connectivity).
        verbose: bool, optional
            If True, print out information about the progress of the algorithm.

        returns a tuple (solution, unrounded, upper_bound) where
        solution: the (rounded) solution w \in {0,1}^m |w| = k
        unrounded: the solution obtained prior to rounding
        upper_bound: the value of the dual at the last iteration

        """

        if k >= len(self.weights):
            # If the budget is larger than the number of candidate edges, then
            # keep them all.
            result = np.ones(len(self.weights))
            if return_rounding_time:
                return result, result, self.evaluate_objective(np.ones(len(self.weights))), 0.0

            return result, result, self.evaluate_objective(np.ones(len(self.weights)))

        # TODO handle case where x is none
        assert(len(x_init) == len(self.weights))

        # Solution for the direction-finding subproblem
        solve_lp = lambda g: constraints.solve_subset_box_lp(g, k)

        # Set up problem to use cache (or not)
        cache = None
        if use_cache:
            cache = MAC.Cache()
        problem = lambda x: self.problem(x, cache=cache)

        # Run Frank-Wolfe to solve the relaxation of subset constrained
        # algebraic connectivity maximization
        w, u = fw.frank_wolfe(initial=x_init, problem=problem,
                                solve_lp=solve_lp, maxiter=max_iters,
                                relative_duality_gap_tol=relative_duality_gap_tol,
                                grad_norm_tol=grad_norm_tol,
                                min_relative_cost_reduction_tol=min_relative_cost_reduction,
                                verbose=verbose, log=self.mac_log)
        start = timer()
        if rounding == "madow":
            rounded = round_madow(w, k, value_fn=self.evaluate_objective, max_iters=random_rounding_max_iters)
        else:
            # rounding == "nearest"
            rounded = round_nearest(w, k, weights=self.weights, break_ties_decimal_tol=10)
        end = timer()
        rounding_time = end - start

        if fallback:
            init_f = self.evaluate_objective(x_init)
            rounded_f = self.evaluate_objective(rounded)

            # If the rounded solution is worse than the initial solution, then
            # return the initial solution instead.
            if rounded_f < init_f:
                rounded = w_init

        # Return the rounded solution along with the unrounded solution and
        # dual upper bound
        if return_rounding_time:
            return rounded, w, u, rounding_time

        return rounded, w, u

    def solve_rie(self, k, x_init=None, rounding="nearest", fallback=False,
              max_iters=5, 
              grad_norm_tol=1e-6, 
              min_relative_cost_reduction=5e-4,
              random_rounding_max_iters=1,
              memory=30,
              verbose=False, return_rounding_time=False, use_cache=False):
        """Use the Frank-Wolfe method to solve the subset selection problem,.

        Parameters
        ----------
        k : int
            Number of edges to select.
        x_init : optional, array-like
            Initial weights for the candidate edges, must satisfy 0 <= w_i <= 1, |w| <= k. This
            is the starting point for the Frank-Wolfe algorithm. TODO(kevin): make optional
        rounding : str, optional
            Rounding method to use. Options are "nearest" (default) and "madow"
            (a random rounding procedure).
        fallback : bool, optional
            If True, fall back to the initialization if the rounded solution is worse.
        max_iters: int, optional
            Maximum number of iterations for the Frank-Wolfe algorithm.
        relative_duality_gap_tol: float, optional
            Tolerance for the relative duality gap, expressed as a fraction of
            the function value. That is, if (upper - f)/f <
            relative_duality_gap_tol, where "upper" is an upper bound on the
            optimal value of 'f', the algorithm terminates.
        grad_norm_tol: float, optional
            Tolerance for the norm of the gradient. If the norm of the gradient
            is less than this value, then the algorithm terminates.
        random_rounding_max_iters: int, optional
            Maximum number of iterations for the random rounding procedure.
            This is only used if rounding="madow". If this is larger than 1,
            then we will randomly round multiple times and return the best
            solution (in terms of algebraic connectivity).
        verbose: bool, optional
            If True, print out information about the progress of the algorithm.

        returns a tuple (solution, unrounded, upper_bound) where
        solution: the (rounded) solution w \in {0,1}^m |w| = k
        unrounded: the solution obtained prior to rounding
        upper_bound: the value of the dual at the last iteration

        """

        # this is a terrible way to handle max iters, but whatever... have fun :)
        self.optimizer._max_iterations = max_iters 
        self.optimizer._min_gradient_norm = grad_norm_tol
        self.optimizer._min_relative_cost_reduction = min_relative_cost_reduction
        self.optimizer.memory = memory
        if verbose:
            self.optimizer._verbosity = 2
        else:
            self.optimizer._verbosity = 0

        if k >= len(self.weights):
            # If the budget is larger than the number of candidate edges, then
            # keep them all.
            result = np.ones(len(self.weights))
            if return_rounding_time:
                return result, result, self.evaluate_objective(np.ones(len(self.weights))), 0.0

            return result, result, self.evaluate_objective(np.ones(len(self.weights)))

        # TODO handle case where x is none
        assert(len(x_init) == len(self.weights))

        manifold = BudgetKManifold(len(self.weights), k, eps_tol=1e-6)
        # cost = pymanopt.function.jax(manifold)(lambda x: -self.problem(x)[0])
        # gradF = pymanopt.function.jax(manifold)(lambda x: -self.problem(x)[1])
        # problem = pymanopt.Problem(manifold, cost, euclidean_gradient=gradF)
        
        cache = None
        if use_cache:
            cache = MAC.Cache()
        problem = lambda x: self.problem_rie(x, cache=cache)

        # problem = lambda x: self.problem(x, cache=None)

        # x_init = manifold.center_pt()

        # x_init = manifold.project_to_feasible_set(x_init)
        # Run Riemannian Conjugate Gradient to solve the relaxation of subset constrained
        # algebraic connectivity maximization
        # breakpoint()
        self.optimizer.reset_step_size_searcher()
        result = self.optimizer.run(manifold, problem, initial_point=manifold.project_to_feasible_set(x_init))
        w = result.point
        u =  result.cost


        start = timer()
        if rounding == "madow":
            rounded = round_madow(w, k, value_fn=self.evaluate_objective, max_iters=random_rounding_max_iters)
        else:
            # rounding == "nearest"
            rounded = round_nearest(w, k, weights=self.weights, break_ties_decimal_tol=10)
        end = timer()
        rounding_time = end - start

        if fallback:
            init_f = self.evaluate_objective(x_init)
            rounded_f = self.evaluate_objective(rounded)

            # If the rounded solution is worse than the initial solution, then
            # return the initial solution instead.
            if rounded_f < init_f:
                rounded = w_init

        # Return the rounded solution along with the unrounded solution and
        # dual upper bound
        if return_rounding_time:
            return rounded, w, u, rounding_time

        return rounded, w, u

class MACH:
    # Defines a hybrid rie/fw solver
    @dataclass
    class Cache:
        """Problem data cache"""
        Q: Optional[np.ndarray] = None

    def __init__(self, fixed_edges, candidate_edges, num_nodes,
                fiedler_method='tracemin_lu', fiedler_tol=1e-8,
                min_selection_weight_tol=1e-10,
                rie_alg="rbfgs"):
        """Parameters
        ----------
        fixed_edges : list of Edge
            List of edges that are fixed in the graph.
        candidate_edges : list of Edge
            List of edges that are candidates for addition to the graph.
        num_nodes : int
            Number of nodes in the graph.
        fiedler_method : str, optional
            Method to use for computing the Fiedler vector. Options are
            'tracemin_lu', 'tracemin_cholesky'. Default is 'tracemin_lu'. Using the
            'tracemin_cholesky' method is faster but requires SuiteSparse to be
            installed.
        fiedler_tol : float, optional
            Tolerance for computing the Fiedler vector and corresponding eigenvalue.
        min_edge_selection_tol : float, optional
            Tolerance for the minimum edge selection weight. Default is 1e-10.
        """
        # Check that we at least *could* have a spanning tree in the set
        # {fixed_edges U candidate_edges} This does not guarantee that a
        # spanning tree exists, but it's a good basic test.
        num_edges = len(fixed_edges) + len(candidate_edges)
        assert (num_nodes - 1) <= num_edges

        # We also check that there aren't "too many" edges. The number of edges
        # in the complete graph K(n) is equal to n * (n - 1) / 2, so we cannot
        # possibly have more edges than this.
        assert num_edges <= 0.5 * num_nodes * (num_nodes - 1)

        # Pre-compute the Laplacian for the subgraph comprised of the "fixed edges".
        self.L_fixed = weight_graph_lap_from_edge_list(fixed_edges, num_nodes)
        self.num_nodes = num_nodes

        self.weights = []
        self.edge_list = []
        for edge in candidate_edges:
            self.weights.append(edge.weight)
            self.edge_list.append((edge.i, edge.j))

        self.weights = np.array(self.weights)
        self.edge_list = np.array(self.edge_list)

        # Configuration for Fiedler vector computation
        self.fiedler_method = fiedler_method
        self.fiedler_tol = fiedler_tol

        # Truncate edges with selection weights below this threshold
        self.min_selection_weight_tol = min_selection_weight_tol
        # self.optimizer = pymanopt.optimizers.ConjugateGradient(verbosity=2, max_iterations=100, min_step_size=1e-3)
        # self.optimizer = pymanopt.optimizers.SteepestDescent(verbosity=2, max_iterations=100, min_step_size=1e-3)

        if rie_alg == "rbfgs":
            self.optimizer = rlbfgs(min_step_size=0, log_verbosity=2, memory=30, step_size_searcher=ConstantAlphaLineSearcher(alpha=1.0),  min_relative_cost_reduction=5e-4)
            # self.optimizer = rlbfgs(min_step_size=1e-5, log_verbosity=2, memory=30, step_size_searcher=ConstantAlphaLineSearcher(alpha=1.0),  min_relative_cost_reduction=5e-4)
        elif rie_alg == "rgd":
            self.optimizer = SteepestDescent(log_verbosity=2, step_size_searcher=ConstantAlphaLineSearcher(alpha=1.00), min_step_size=1e-4, min_relative_cost_reduction=1e-8)
            # self.optimizer = SteepestDescent(log_verbosity=2, step_size_searcher=ConstantAlphaLineSearcher(alpha=1.00), min_step_size=1e-4, min_relative_cost_reduction=1e-8)
        else:
            raise ValueError(f"Unknown Riemannian optimization algorithm: {rie_alg}. Choose 'rbfgs' or 'rgd'.")

        self.prob_run_info = {'runtimes':[], 'calls': 0, 'num_idx':[], 'lap_rt':[],'weight_graph_count':0, 'wgle_rt':[], 'fiedler_iters':[], 'fiedler_runtimes':[], 'ratio':[]}
        self.prob_rie_run_info = {'runtimes':[], 'calls': 0, 'num_idx':[], 'lap_rt':[],'weight_graph_count':0, 'wgle_rt':[], 'fiedler_iters':[], 'fiedler_runtimes':[], 'ratio':[]}

        self.mac_log = {"iterations": {"iteration": [], "time": [], "point": [], "cost": [], "gradient_norm": []}}

    def reset_log(self):
        self.optimizer.reset_log()
        self.mac_log = {"iterations": {"iteration": [], "time": [], "point": [], "cost": [], "gradient_norm": []}}
        # self.prob_run_info = {'runtimes':[], 'calls': 0, 'num_idx':[], 'lap_rt':[], 'weight_graph_count':0, 'wgle_rt':[]}
        # self.prob_rie_run_info = {'runtimes':[], 'calls': 0, 'num_idx':[], 'lap_rt':[],'weight_graph_count':0, 'wgle_rt':[]}

        self.prob_run_info = {'runtimes':[], 'calls': 0, 'num_idx':[], 'lap_rt':[],'weight_graph_count':0, 'wgle_rt':[], 'fiedler_iters':[], 'fiedler_runtimes':[], 'ratio':[]}
        self.prob_rie_run_info = {'runtimes':[], 'calls': 0, 'num_idx':[], 'lap_rt':[],'weight_graph_count':0, 'wgle_rt':[], 'fiedler_iters':[], 'fiedler_runtimes':[], 'ratio':[]}

    def laplacian(self, x, solver='fw'):
        """Construct the combined Laplacian (fixed edges plus candidate edges weighted by x).
        x: An element of [0,1]^m; this is the edge selection to use

        The tolerance parameter `min_selection_weight_tol` is used to prune out
        edges that are numerically zero. This improves speed in situations
        where edges are not *exactly* zero, but close enough that they have
        almost no influence on the graph.

        returns the matrix L(x)
        """
        start = timer()
        idx = np.where(x > self.min_selection_weight_tol)
        prod = x[idx]*self.weights[idx]
        start2 = timer()
        L_candidate = weight_graph_lap_from_edges(self.edge_list[idx], prod, self.num_nodes)
        end2 = timer()
        L_x = self.L_fixed + L_candidate
        end = timer()
        if solver == 'fw':
            self.prob_run_info['lap_rt'].append(end - start)
            self.prob_run_info['num_idx'].append(len(idx[0]))
            self.prob_run_info['wgle_rt'].append(end2-start2)
        else:
            self.prob_rie_run_info['lap_rt'].append(end - start)
            self.prob_rie_run_info['num_idx'].append(len(idx[0]))
            self.prob_rie_run_info['wgle_rt'].append(end2-start2)
        return L_x

    def evaluate_objective(self, x):
        """
        Compute lambda_2(L(x)) where L(x) is the Laplacian with edge i weighted
        by x_i*weight_i and lambda_2 is the second smallest eigenvalue (this is the
        algebraic connectivity).

        x: Weights for each candidate edge (does not include fixed edges)

        returns F(x) = lambda_2(L(x)).
        """
        return fiedler.find_fiedler_pair(L=self.laplacian(x),
                                         method=self.fiedler_method, tol=self.fiedler_tol)[0]

    def problem(self, x, cache=None, rie=False):
        """Compute the algebraic connectivity of L(x) and a (super)gradient of the
        algebraic connectivity with respect to x.

        x: Weights for each candidate edge (does not include fixed edges)
        cache: Mutable `Cache` object. If a `Cache` object is provided in the `cache` field, it will be used
        and updated, but not explicitly returned. Rather, it will be updated directly.

        returns x, grad F(x).
        """
        start = timer()
        Q = None if cache is None else cache.Q
        # f, fiedler_vec, Qnew = fiedler.find_fiedler_pair(L=self.laplacian(x,solver='fw'), X=Q, method='tracemin_cholesky')
        (f, fiedler_vec, Qnew), (average_time, iters,  ratio) = fiedler.find_fiedler_pair(L=self.laplacian(x, solver='fw'), X=Q, method='tracemin_cholesky')
        end = timer()
        # f, fiedler_vec, Qnew = fiedler.find_fiedler_pair(L=self.laplacian(x, solver='rie'), X=Q, method='tracemin_pcg')
        gradf = np.zeros(len(self.weights))
        for k in range(len(self.weights)):
            edge = self.edge_list[k] # get edge (i,j)
            v_i = fiedler_vec[edge[0]]
            v_j = fiedler_vec[edge[1]]
            weight_k = self.weights[k]
            kdelta = weight_k * (v_i - v_j)
            gradf[k] = kdelta * (v_i - v_j)

        if cache is not None:
            cache.Q = Q

        # end = timer()
        self.prob_run_info['calls'] += 1
        self.prob_run_info['runtimes'].append(end - start)
        self.prob_run_info['fiedler_iters'].append(iters)
        self.prob_run_info['fiedler_runtimes'].append(average_time)
        self.prob_run_info['ratio'].append(ratio)
        if rie:
            return -f, -gradf
        else:
            return f, gradf

    def solve(self, k, x_init=None, rounding="nearest", fallback=False,
              max_iters=5, relative_duality_gap_tol=1e-4, min_relative_cost_reduction=5e-4,
              grad_norm_tol=1e-8, random_rounding_max_iters=1,
              verbose=False, return_rounding_time=False, use_cache=False):
        """Use the Riemannian and Frank-Wolfe method to solve the subset selection problem,.

        Parameters
        ----------
        k : int
            Number of edges to select.
        x_init : optional, array-like
            Initial weights for the candidate edges, must satisfy 0 <= w_i <= 1, |w| <= k. This
            is the starting point for the Frank-Wolfe algorithm. TODO(kevin): make optional
        rounding : str, optional
            Rounding method to use. Options are "nearest" (default) and "madow"
            (a random rounding procedure).
        fallback : bool, optional
            If True, fall back to the initialization if the rounded solution is worse.
        max_iters: int, optional
            Maximum number of iterations for the Frank-Wolfe algorithm.
        relative_duality_gap_tol: float, optional
            Tolerance for the relative duality gap, expressed as a fraction of
            the function value. That is, if (upper - f)/f <
            relative_duality_gap_tol, where "upper" is an upper bound on the
            optimal value of 'f', the algorithm terminates.
        grad_norm_tol: float, optional
            Tolerance for the norm of the gradient. If the norm of the gradient
            is less than this value, then the algorithm terminates.
        random_rounding_max_iters: int, optional
            Maximum number of iterations for the random rounding procedure.
            This is only used if rounding="madow". If this is larger than 1,
            then we will randomly round multiple times and return the best
            solution (in terms of algebraic connectivity).
        verbose: bool, optional
            If True, print out information about the progress of the algorithm.

        returns a tuple (solution, unrounded, upper_bound) where
        solution: the (rounded) solution w \in {0,1}^m |w| = k
        unrounded: the solution obtained prior to rounding
        upper_bound: the value of the dual at the last iteration

        """

        if k >= len(self.weights):
            # If the budget is larger than the number of candidate edges, then
            # keep them all.
            result = np.ones(len(self.weights))
            if return_rounding_time:
                return result, result, self.evaluate_objective(np.ones(len(self.weights))), 0.0

            return result, result, self.evaluate_objective(np.ones(len(self.weights)))

        # TODO handle case where x is none
        assert(len(x_init) == len(self.weights))

        # Solution for the direction-finding subproblem
        solve_lp = lambda g: constraints.solve_subset_box_lp(g, k)

        # Set up problem to use cache (or not)
        cache = None
        if use_cache:
            cache = MAC.Cache()

        problem_rie = lambda x: self.problem(x, cache=cache, rie=True)
        problem = lambda x: self.problem(x, cache=cache)

        self.optimizer._max_iterations = 5
        self.optimizer._min_gradient_norm = grad_norm_tol
        self.optimizer._min_relative_cost_reduction = min_relative_cost_reduction
        self.optimizer.memory = 5

        manifold = BudgetKManifold(len(self.weights), k, eps_tol=1e-6)
        x_init = manifold.center_pt()
        self.optimizer.reset_step_size_searcher()

        result = self.optimizer.run(manifold, problem_rie, initial_point=x_init)
        fw_init = result.point

        self.mac_log = self.optimizer._log
        # make entries of mac_log iterations cost negative of their current value
        self.mac_log["iterations"]["cost"] = [-x for x in self.mac_log["iterations"]["cost"]]
        # Run Frank-Wolfe to solve the relaxation of subset constrained
        # algebraic connectivity maximization
        rie_iters = len(self.mac_log["iterations"]["iteration"])
        stepsize = lambda x, g, s, k: 2.0 / (rie_iters + k) # constant step size of 2 / num iterations
        w, u = fw.frank_wolfe(initial=fw_init, problem=problem,
                                solve_lp=solve_lp, maxiter=max_iters-rie_iters,
                                stepsize=stepsize,
                                relative_duality_gap_tol=relative_duality_gap_tol,
                                grad_norm_tol=grad_norm_tol,
                                min_relative_cost_reduction_tol=min_relative_cost_reduction,
                                verbose=verbose, log=self.mac_log)

        start = timer()
        if rounding == "madow":
            rounded = round_madow(w, k, value_fn=self.evaluate_objective, max_iters=random_rounding_max_iters)
        else:
            # rounding == "nearest"
            rounded = round_nearest(w, k, weights=self.weights, break_ties_decimal_tol=10)
        end = timer()
        rounding_time = end - start

        if fallback:
            init_f = self.evaluate_objective(x_init)
            rounded_f = self.evaluate_objective(rounded)

            # If the rounded solution is worse than the initial solution, then
            # return the initial solution instead.
            if rounded_f < init_f:
                rounded = x_init

        # Return the rounded solution along with the unrounded solution and
        # dual upper bound
        if return_rounding_time:
            return rounded, w, u, rounding_time

        return rounded, w, u


class MACHR:
    @dataclass
    class Cache:
        """Problem data cache"""
        Q: Optional[np.ndarray] = None

    def __init__(self, fixed_edges, candidate_edges, num_nodes,
                fiedler_method='tracemin_lu', fiedler_tol=1e-8,
                min_selection_weight_tol=1e-10,
                rie_alg="rbfgs"):
        """Parameters
        ----------
        fixed_edges : list of Edge
            List of edges that are fixed in the graph.
        candidate_edges : list of Edge
            List of edges that are candidates for addition to the graph.
        num_nodes : int
            Number of nodes in the graph.
        fiedler_method : str, optional
            Method to use for computing the Fiedler vector. Options are
            'tracemin_lu', 'tracemin_cholesky'. Default is 'tracemin_lu'. Using the
            'tracemin_cholesky' method is faster but requires SuiteSparse to be
            installed.
        fiedler_tol : float, optional
            Tolerance for computing the Fiedler vector and corresponding eigenvalue.
        min_edge_selection_tol : float, optional
            Tolerance for the minimum edge selection weight. Default is 1e-10.
        """
        # Check that we at least *could* have a spanning tree in the set
        # {fixed_edges U candidate_edges} This does not guarantee that a
        # spanning tree exists, but it's a good basic test.
        num_edges = len(fixed_edges) + len(candidate_edges)
        assert (num_nodes - 1) <= num_edges

        # We also check that there aren't "too many" edges. The number of edges
        # in the complete graph K(n) is equal to n * (n - 1) / 2, so we cannot
        # possibly have more edges than this.
        assert num_edges <= 0.5 * num_nodes * (num_nodes - 1)

        # Pre-compute the Laplacian for the subgraph comprised of the "fixed edges".
        self.L_fixed = weight_graph_lap_from_edge_list(fixed_edges, num_nodes)
        self.num_nodes = num_nodes

        self.weights = []
        self.edge_list = []
        for edge in candidate_edges:
            self.weights.append(edge.weight)
            self.edge_list.append((edge.i, edge.j))

        self.weights = np.array(self.weights)
        self.edge_list = np.array(self.edge_list)

        # Configuration for Fiedler vector computation
        self.fiedler_method = fiedler_method
        self.fiedler_tol = fiedler_tol

        # Truncate edges with selection weights below this threshold
        self.min_selection_weight_tol = min_selection_weight_tol
        # self.optimizer = pymanopt.optimizers.ConjugateGradient(verbosity=2, max_iterations=100, min_step_size=1e-3)
        # self.optimizer = pymanopt.optimizers.SteepestDescent(verbosity=2, max_iterations=100, min_step_size=1e-3)

        if rie_alg == "rbfgs":
            self.optimizer = rlbfgs(min_step_size=1e-5, log_verbosity=2, memory=30, step_size_searcher=ConstantAlphaLineSearcher(alpha=1.0),  min_relative_cost_reduction=5e-4)
        elif rie_alg == "rgd":
            self.optimizer = SteepestDescent(log_verbosity=2, step_size_searcher=ConstantAlphaLineSearcher(alpha=1.00), min_step_size=1e-4, min_relative_cost_reduction=1e-8)
        else:
            raise ValueError(f"Unknown Riemannian optimization algorithm: {rie_alg}. Choose 'rbfgs' or 'rgd'.")

        self.prob_run_info = {'runtimes':[], 'calls': 0, 'num_idx':[], 'lap_rt':[],'weight_graph_count':0, 'wgle_rt':[], 'fiedler_iters':[], 'fiedler_runtimes':[], 'ratio':[]}
        self.prob_rie_run_info = {'runtimes':[], 'calls': 0, 'num_idx':[], 'lap_rt':[],'weight_graph_count':0, 'wgle_rt':[], 'fiedler_iters':[], 'fiedler_runtimes':[], 'ratio':[]}

        self.mac_log = {"iterations": {"iteration": [], "time": [], "point": [], "cost": [], "gradient_norm": []}}

    def reset_log(self):
        self.optimizer.reset_log()
        self.mac_log = {"iterations": {"iteration": [], "time": [], "point": [], "cost": [], "gradient_norm": []}}
        # self.prob_run_info = {'runtimes':[], 'calls': 0, 'num_idx':[], 'lap_rt':[], 'weight_graph_count':0, 'wgle_rt':[]}
        # self.prob_rie_run_info = {'runtimes':[], 'calls': 0, 'num_idx':[], 'lap_rt':[],'weight_graph_count':0, 'wgle_rt':[]}

        self.prob_run_info = {'runtimes':[], 'calls': 0, 'num_idx':[], 'lap_rt':[],'weight_graph_count':0, 'wgle_rt':[], 'fiedler_iters':[], 'fiedler_runtimes':[], 'ratio':[]}
        self.prob_rie_run_info = {'runtimes':[], 'calls': 0, 'num_idx':[], 'lap_rt':[],'weight_graph_count':0, 'wgle_rt':[], 'fiedler_iters':[], 'fiedler_runtimes':[], 'ratio':[]}

    def laplacian(self, x, solver='fw'):
        """Construct the combined Laplacian (fixed edges plus candidate edges weighted by x).
        x: An element of [0,1]^m; this is the edge selection to use

        The tolerance parameter `min_selection_weight_tol` is used to prune out
        edges that are numerically zero. This improves speed in situations
        where edges are not *exactly* zero, but close enough that they have
        almost no influence on the graph.

        returns the matrix L(x)
        """
        start = timer()
        idx = np.where(x > self.min_selection_weight_tol)
        prod = x[idx]*self.weights[idx]
        start2 = timer()
        L_candidate = weight_graph_lap_from_edges(self.edge_list[idx], prod, self.num_nodes)
        end2 = timer()
        L_x = self.L_fixed + L_candidate
        end = timer()
        if solver == 'fw':
            self.prob_run_info['lap_rt'].append(end - start)
            self.prob_run_info['num_idx'].append(len(idx[0]))
            self.prob_run_info['wgle_rt'].append(end2-start2)
        else:
            self.prob_rie_run_info['lap_rt'].append(end - start)
            self.prob_rie_run_info['num_idx'].append(len(idx[0]))
            self.prob_rie_run_info['wgle_rt'].append(end2-start2)
        return L_x

    def evaluate_objective(self, x):
        """
        Compute lambda_2(L(x)) where L(x) is the Laplacian with edge i weighted
        by x_i*weight_i and lambda_2 is the second smallest eigenvalue (this is the
        algebraic connectivity).

        x: Weights for each candidate edge (does not include fixed edges)

        returns F(x) = lambda_2(L(x)).
        """
        return fiedler.find_fiedler_pair(L=self.laplacian(x),
                                         method=self.fiedler_method, tol=self.fiedler_tol)[0]

    def problem(self, x, cache=None, rie=False):
        """Compute the algebraic connectivity of L(x) and a (super)gradient of the
        algebraic connectivity with respect to x.

        x: Weights for each candidate edge (does not include fixed edges)
        cache: Mutable `Cache` object. If a `Cache` object is provided in the `cache` field, it will be used
        and updated, but not explicitly returned. Rather, it will be updated directly.

        returns x, grad F(x).
        """
        start = timer()
        Q = None if cache is None else cache.Q
        # f, fiedler_vec, Qnew = fiedler.find_fiedler_pair(L=self.laplacian(x,solver='fw'), X=Q, method='tracemin_cholesky')
        (f, fiedler_vec, Qnew), (average_time, iters,  ratio) = fiedler.find_fiedler_pair(L=self.laplacian(x, solver='fw'), X=Q, method='tracemin_cholesky')
        end = timer()
        # f, fiedler_vec, Qnew = fiedler.find_fiedler_pair(L=self.laplacian(x, solver='rie'), X=Q, method='tracemin_pcg')
        gradf = np.zeros(len(self.weights))
        for k in range(len(self.weights)):
            edge = self.edge_list[k] # get edge (i,j)
            v_i = fiedler_vec[edge[0]]
            v_j = fiedler_vec[edge[1]]
            weight_k = self.weights[k]
            kdelta = weight_k * (v_i - v_j)
            gradf[k] = kdelta * (v_i - v_j)

        if cache is not None:
            cache.Q = Q

        # end = timer()
        self.prob_run_info['calls'] += 1
        self.prob_run_info['runtimes'].append(end - start)
        self.prob_run_info['fiedler_iters'].append(iters)
        self.prob_run_info['fiedler_runtimes'].append(average_time)
        self.prob_run_info['ratio'].append(ratio)
        if rie:
            return -f, -gradf
        else:
            return f, gradf

    def solve(self, k, x_init=None, rounding="nearest", fallback=False,
              max_iters=5, relative_duality_gap_tol=1e-4, min_relative_cost_reduction=5e-4,
              grad_norm_tol=1e-8, random_rounding_max_iters=1,
              verbose=False, return_rounding_time=False, use_cache=False):
        """Use the Riemannian and Frank-Wolfe method to solve the subset selection problem,.

        Parameters
        ----------
        k : int
            Number of edges to select.
        x_init : optional, array-like
            Initial weights for the candidate edges, must satisfy 0 <= w_i <= 1, |w| <= k. This
            is the starting point for the Frank-Wolfe algorithm. TODO(kevin): make optional
        rounding : str, optional
            Rounding method to use. Options are "nearest" (default) and "madow"
            (a random rounding procedure).
        fallback : bool, optional
            If True, fall back to the initialization if the rounded solution is worse.
        max_iters: int, optional
            Maximum number of iterations for the Frank-Wolfe algorithm.
        relative_duality_gap_tol: float, optional
            Tolerance for the relative duality gap, expressed as a fraction of
            the function value. That is, if (upper - f)/f <
            relative_duality_gap_tol, where "upper" is an upper bound on the
            optimal value of 'f', the algorithm terminates.
        grad_norm_tol: float, optional
            Tolerance for the norm of the gradient. If the norm of the gradient
            is less than this value, then the algorithm terminates.
        random_rounding_max_iters: int, optional
            Maximum number of iterations for the random rounding procedure.
            This is only used if rounding="madow". If this is larger than 1,
            then we will randomly round multiple times and return the best
            solution (in terms of algebraic connectivity).
        verbose: bool, optional
            If True, print out information about the progress of the algorithm.

        returns a tuple (solution, unrounded, upper_bound) where
        solution: the (rounded) solution w \in {0,1}^m |w| = k
        unrounded: the solution obtained prior to rounding
        upper_bound: the value of the dual at the last iteration

        """

        if k >= len(self.weights):
            # If the budget is larger than the number of candidate edges, then
            # keep them all.
            result = np.ones(len(self.weights))
            if return_rounding_time:
                return result, result, self.evaluate_objective(np.ones(len(self.weights))), 0.0

            return result, result, self.evaluate_objective(np.ones(len(self.weights)))

        # TODO handle case where x is none
        assert(len(x_init) == len(self.weights))

        # Solution for the direction-finding subproblem
        solve_lp = lambda g: constraints.solve_subset_box_lp(g, k)

        # Set up problem to use cache (or not)
        cache = None
        if use_cache:
            cache = MAC.Cache()

        problem_rie = lambda x: self.problem(x, cache=cache, rie=True)
        problem = lambda x: self.problem(x, cache=cache)

        stepsize = lambda x, g, s, k: 2.0 / (2 + k) # constant step size of 2 / num iterations

        rie_init, u = fw.frank_wolfe(initial=x_init, problem=problem,
                                solve_lp=solve_lp, maxiter=10,
                                stepsize=stepsize,
                                relative_duality_gap_tol=relative_duality_gap_tol,
                                grad_norm_tol=grad_norm_tol,
                                min_relative_cost_reduction_tol=min_relative_cost_reduction,
                                verbose=verbose, log=self.mac_log)

        self.optimizer._max_iterations = max_iters
        self.optimizer._min_gradient_norm = grad_norm_tol
        self.optimizer._min_relative_cost_reduction = min_relative_cost_reduction
        self.optimizer.memory = max_iters

        manifold = BudgetKManifold(len(self.weights), k, eps_tol=1e-6)
        # x_init = manifold.center_pt()
        self.optimizer.reset_step_size_searcher()
        result = self.optimizer.run(manifold, problem_rie, initial_point=manifold.project_to_feasible_set(rie_init))
        w = result.point

        # self.mac_log["iterations"]["cost"].extend(self.optimizer._log["iterations"]['cost'])
        self.mac_log["iterations"]["cost"].extend([-x for x in self.optimizer._log["iterations"]["cost"]])
        self.mac_log["iterations"]["iteration"].extend(self.optimizer._log["iterations"]['iteration'])
        self.mac_log["iterations"]["time"].extend(self.optimizer._log["iterations"]['time'])
        self.mac_log["iterations"]["point"].extend(self.optimizer._log["iterations"]['point'])
        self.mac_log["iterations"]["gradient_norm"].extend(self.optimizer._log["iterations"]['gradient_norm'])

        # make entries of mac_log iterations cost negative of their current value
        # Run Frank-Wolfe to solve the relaxation of subset constrained
        # algebraic connectivity maximization

        start = timer()
        if rounding == "madow":
            rounded = round_madow(w, k, value_fn=self.evaluate_objective, max_iters=random_rounding_max_iters)
        else:
            # rounding == "nearest"
            rounded = round_nearest(w, k, weights=self.weights, break_ties_decimal_tol=10)
        end = timer()
        rounding_time = end - start

        if fallback:
            init_f = self.evaluate_objective(x_init)
            rounded_f = self.evaluate_objective(rounded)

            # If the rounded solution is worse than the initial solution, then
            # return the initial solution instead.
            if rounded_f < init_f:
                rounded = x_init

        # Return the rounded solution along with the unrounded solution and
        # dual upper bound
        if return_rounding_time:
            return rounded, w, u, rounding_time

        return rounded, w, u