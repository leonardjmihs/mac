import numpy as np

class BackTrackingLineSearcher:
    """Back-tracking line-search algorithm."""

    def __init__(
        self,
        contraction_factor=0.5,
        optimism=2,
        sufficient_decrease=1e-4,
        max_iterations=25,
        initial_step_size=1,
    ):
        self.contraction_factor = contraction_factor
        self.optimism = optimism
        self.sufficient_decrease = sufficient_decrease
        self.max_iterations = max_iterations
        self.initial_step_size = initial_step_size

        self._oldf0 = None

    def reset(self):
        self._oldf0 = None

    def search(self, problem, manifold, x, d, f0, df0, iter=0):
        """Function to perform backtracking line search.

        Args:
            objective: Objective function to optimize.
            manifold: The manifold to optimize over.
            x: Starting point on the manifold.
            d: Tangent vector at ``x``, i.e., a descent direction.
            df0: Directional derivative at ``x`` along ``d``.

        Returns:
            A tuple ``(step_size, newx)`` where ``step_size`` is the norm of
            the vector retracted to reach the suggested iterate ``newx`` from
            ``x``.
        """
        # Compute the norm of the search direction
        norm_d = manifold.norm(x, d)

        if self._oldf0 is not None:
            # Pick initial step size based on where we were last time.
            alpha = 2 * (f0 - self._oldf0) / df0
            # Look a little further
            alpha *= self.optimism
        else:
            alpha = self.initial_step_size / norm_d
        alpha = float(alpha)

        # Make the chosen step and compute the cost there.
        newx = manifold.retraction(x, alpha * d)
        newf, _ = problem(np.array(newx))
        step_count = 1

        # Backtrack while the Armijo criterion is not satisfied
        while (
            newf > f0 + self.sufficient_decrease * alpha * df0
            and step_count <= self.max_iterations
        ):

            # Reduce the step size
            alpha = self.contraction_factor * alpha

            # and look closer down the line
            newx = manifold.retraction(x, alpha * d)
            newf, _ = problem(np.array(newx))

            step_count = step_count + 1

        # If we got here without obtaining a decrease, we reject the step.
        if newf > f0:
            alpha = 0
            newx = x

        step_size = alpha * norm_d

        self._oldf0 = f0

        return step_size, newx

class AdaptiveLineSearcher:
    """Adaptive line-search algorithm."""

    def __init__(
        self,
        contraction_factor=0.5,
        sufficient_decrease=0.5,
        max_iterations=10,
        initial_step_size=1,
    ):
        self._contraction_factor = contraction_factor
        self._sufficient_decrease = sufficient_decrease
        self._max_iterations = max_iterations
        self._initial_step_size = initial_step_size
        self._oldalpha = None

    def reset(self):
        self._oldalpha = None

    def search(self, problem, manifold, x, d, f0, df0, iter=0):
        norm_d = manifold.norm(x, d)

        if self._oldalpha is not None:
            alpha = self._oldalpha
        else:
            alpha = self._initial_step_size / norm_d
        alpha = float(alpha)

        newx = manifold.retraction(x, alpha * d)
        newf, _ = problem(np.array(newx))
        cost_evaluations = 1

        while (
            newf > f0 + self._sufficient_decrease * alpha * df0
            and cost_evaluations <= self._max_iterations
        ):
            # Reduce the step size.
            alpha *= self._contraction_factor

            # Look closer down the line.
            newx = manifold.retraction(x, alpha * d)
            newf,_= problem(np.array(newx))

            cost_evaluations += 1

        if newf > f0:
            alpha = 0
            newx = x

        step_size = alpha * norm_d

        # Store a suggestion for what the next initial step size trial should
        # be. On average we intend to do only one extra cost evaluation. Notice
        # how the suggestion is not about step_size but about alpha. This is
        # the reason why this line search is not invariant under rescaling of
        # the search direction d.

        # If things go reasonably well, try to keep pace.
        if cost_evaluations == 2:
            self._oldalpha = alpha
        # If things went very well or we backtracked a lot (meaning the step
        # size is probably quite small), speed up.
        else:
            self._oldalpha = 2 * alpha

        return step_size, newx

class NaieveLineSearcher:
    """Adaptive line-search algorithm."""

    def __init__(
        self,
        optimism=2,
        initial_step_size=1,
    ):
        self._optimism = optimism
        self._initial_step_size = initial_step_size

        self._oldf0 = None

    def reset(self):
        self._oldf0 = None

    def search(self, problem, manifold, x, d, f0, df0, iter=0):
        norm_d = manifold.norm(x, d)

        if self._oldf0 is not None:
            # alpha = 2 * (f0 - self._oldf0) / df0
            alpha = 2 * (f0 - self._oldf0) / self._oldf0
            # Look a little further
            alpha *= self._optimism
        else:
            alpha = self._initial_step_size / norm_d
        alpha = float(abs(alpha))

        step_size = alpha * norm_d

        newx = manifold.retraction(x, alpha * d)
        self._oldf0 = f0

        return step_size, newx

class ConstantAlphaLineSearcher:
    """Constant Alpha."""

    def __init__(
        self,
        alpha=1.0
    ):
        self.alpha = alpha

    def reset(self):
        pass

    def search(self, problem, manifold, x, d, f0, df0, iter=0):
        alpha = self.alpha
        newx = manifold.retraction(x, alpha * d)
        step_size = alpha * manifold.norm(x, d)
        return step_size, newx

class ConstantStepSizeLineSearcher:
    """Constant Step Size"""

    def __init__(
        self,
        step_size=1.0
    ):
        self.step_size = step_size

    def reset(self):
        pass

    def search(self, problem, manifold, x, d, f0, df0, iter=0):
        alpha = self.step_size / manifold.norm(x, d)
        newx = manifold.retraction(x, alpha * d)
        return self.step_size, newx 

class ShrinkingStepSizeLineSearcher:
    """Adaptive line-search algorithm."""

    def __init__(
        self,
        step_size=1.0,
        decay_rate=2.0
    ):
        self._starting_step_size = step_size
        self._decay_rate = decay_rate

    def reset(self):
        pass

    def search(self, problem, manifold, x, d, f0, df0, iter):
        step_size = iter / (iter+self._decay_rate) * self._starting_step_size
        alpha = step_size / manifold.norm(x, d)
        newx = manifold.retraction(x, alpha * d)
        return step_size, newx

class ShrinkingAlphaLineSearcher:
    """Adaptive line-search algorithm."""

    def __init__(
        self,
        alpha=1.0,
        decay_rate=2.0
    ):
        self._starting_alpha = alpha
        self._decay_rate = decay_rate

    def reset(self):
        pass

    def search(self, problem, manifold, x, d, f0, df0, iter):
        norm_d = manifold.norm(x, d)
        alpha = iter / (iter+self._decay_rate) * self._starting_alpha
        step_size = alpha * manifold.norm(x, d)
        newx = manifold.retraction(x, alpha * d)
        return step_size, newx

# class ShrinkingLineSearcher:
#     """Adaptive line-search algorithm."""

#     def __init__(
#         self,
#         step_size=1.0
#         # alpha=1.0
#     ):
#         self._starting_step_size = step_size

#     def reset(self):
#         pass

#     def search(self, problem, manifold, x, d, f0, df0, iter):
#         norm_d = manifold.norm(x, d)
#         alpha = iter / (iter+2.0) * self._starting_step_size
#         alpha = float(alpha)
#         step_size = alpha * norm_d
#         newx = manifold.retraction(x, alpha * d)
#         return step_size, newx