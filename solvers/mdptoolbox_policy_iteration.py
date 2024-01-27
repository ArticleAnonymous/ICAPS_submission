"""
Solver calling the MDP Toolbox Policy Iteration solver.
"""

from mdptoolbox.mdp import PolicyIteration
import numpy as np
from utils.generic_model import GenericModel
import time


class Solver:
    def __init__(self, env: GenericModel, discount: float, max_iter: int = int(1e8)):
        self.env = env
        self.discount = discount
        self.max_iter = max_iter
        self.name = "mdptoolbox_policy_iteration"

        self.value = None
        self.policy = None

    def run(self):
        start_time = time.time()
        self.model = PolicyIteration(
            self.env.transition_matrix,
            self.env.reward_matrix,
            discount=self.discount,
            max_iter=self.max_iter,
            eval_type="iterative",
            skip_check=True,
        )
        self.model.run()
        self.runtime = time.time() - start_time

        self.value = np.array(self.model.V)
        self.policy = np.array(self.model.policy)
