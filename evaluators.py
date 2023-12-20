import numpy as np
from abc import abstractmethod


class Evaluator:
    """
    Abstract class for evaluators.
    This class also mantains the registry for the evaluators.
    """
    registry = {}

    def __init_subclass__(cls, **kwargs):
        """When initializing a subclass, add it to the registry."""
        Evaluator.registry[cls.__name__] = cls

    @classmethod
    def get_class(cls, name):
        return Evaluator.registry[name]

    def __call__(self, program):
        """Evaluate the program and return its loss."""
        return self.evaluate(program)

    def evaluate(self, program):
        """Evaluate the program and return its loss."""
        try:
            print(f"Testing {program}")
            exec(program, globals(), locals())
            heuristic = locals()['solution']
            return self._execute_and_return_loss(heuristic)
        except:
            # Malformed solution
            return float("inf")

    @abstractmethod
    def _execute_and_return_loss(self, heuristic):
        pass


class OneMaxEvaluator(Evaluator):
    """
    Trivial problem, just for testing.
    The goal of this problem is to maximize the number of ones in an array.
    """

    def __init__(self, dim):
        self._dim = dim

    def _execute_and_return_loss(self, heuristic):
        cur_solution = []
        for _ in range(self._dim):
            # Evaluate the possible choices at each step and choose greedily
            choices = [0, 1]
            scores = []
            for choice in choices:
                scores.append(heuristic(cur_solution + [choice]))
            best = np.argmax(scores)
            cur_solution.append(choices[best])
        # The goal of this problem is to maximize the number of ones, but we're minimizing, so I return the opposite
        return -sum(cur_solution)


class OddMaxEvaluator(Evaluator):
    """
    Trivial problem, just for testing.
    The goal of this problem is to maximize the number of ones in the odd positions of an array, while minimizing the number of ones in the even positions.
    """

    def __init__(self, dim):
        self._dim = dim

    def _execute_and_return_loss(self, heuristic):
        cur_solution = []
        for _ in range(self._dim):
            # Evaluate the possible choices at each step and choose greedily
            choices = [0, 1]
            scores = []
            for choice in choices:
                scores.append(heuristic(cur_solution + [choice]))
            best = np.argmax(scores)
            cur_solution.append(choices[best])
        # The goal of this problem is to maximize the number of ones, but we're minimizing, so I return the opposite
        return -sum(cur_solution[1::2]) + sum(cur_solution[::2])


