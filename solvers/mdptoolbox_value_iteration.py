"""
Solver calling the MDP Toolbox Value Iteration solver.
"""

from mdptoolbox.mdp import ValueIteration
import numpy as np
from utils.generic_model import GenericModel
import time


class Solver:
    def __init__(
        self,
        model: GenericModel,
        discount: float,
        epsilon: float = 1e-4,
        max_iter: int = int(1e8),
    ):
        self.model = model
        self.discount = discount
        self.epsilon = epsilon
        self.name = "mdptoolbox_value_iteration"
        self.transition_matrix_is_sparse: bool = not isinstance(
            self.model.transition_matrix, np.ndarray
        )
        self.max_iter = max_iter

        self.value: np.ndarray = None
        self.policy: np.ndarray = None

    def run(self):
        start_time = time.time()
        self.model = ValueIteration(
            self.model.transition_matrix,
            self.model.reward_matrix,
            discount=self.discount,
            epsilon=self.epsilon,
            max_iter=self.max_iter,
            skip_check=True,
        )
        self.model.run()
        self.runtime = time.time() - start_time

        self.value = np.array(self.model.V)
        self.policy = np.array(self.model.policy)
