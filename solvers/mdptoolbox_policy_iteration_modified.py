"""
Solver calling the MDP Toolbox Modified Policy Iteration solver.
"""

from mdptoolbox.mdp import PolicyIterationModified
import numpy as np
from utils.generic_model import GenericModel
import time


class Solver:
    def __init__(
        self,
        env: GenericModel,
        discount: float,
        epsilon: float = 1e-4,
        max_iter_policy_update: int = int(1e8),
    ):
        self.env = env
        self.gamma = discount
        self.epsilon = epsilon
        self.max_iter = max_iter_policy_update

        self.name = "mdptoolbox_policy_iteration_modified"
        self.value = None
        self.policy = None

    def run(self):
        start_time = time.time()

        self.model = PolicyIterationModified(
            self.env.transition_matrix,
            self.env.reward_matrix,
            discount=self.gamma,
            epsilon=self.epsilon,
            max_iter=self.max_iter,
            skip_check=True,
        )
        self.model.run()
        self.runtime = time.time() - start_time

        self.value = np.array(self.model.V)
        self.policy = np.array(self.model.policy)
