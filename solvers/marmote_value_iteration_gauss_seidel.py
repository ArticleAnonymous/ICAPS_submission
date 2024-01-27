"""
Solver calling the Marmote Value Iteration Gauss-Seidel solver in C++.
"""

from solvers.pyMarmoteMDP import (
    marmoteInterval,
    sparseMatrix,
    discountedMDP,
    solutionMDP,
)
from time import time
import numpy as np
from utils.generic_model import GenericModel


class Solver:
    def __init__(
        self,
        env: GenericModel,
        discount: float,
        epsilon: float = 1e-4,
        max_iter: int = int(1e8),
    ):
        self.env = env
        self.discount = discount
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.name = "marmote_value_iteration_gauss_seidel"
        self.transition_matrix_is_sparse: bool = not isinstance(
            self.env.transition_matrix, np.ndarray
        )

    def _build_reward_matrix(self):
        self.reward_matrix = sparseMatrix(self.S, self.A)
        for a in range(self.env.action_dim):
            for s in range(self.env.state_dim):
                self.reward_matrix.addToEntry(s, a, self.env.reward_matrix[s, a])

    def _build_transition_matrix(self):
        self.transitions_list = list()
        for a in range(self.env.action_dim):
            P = sparseMatrix(self.S)
            for s1 in range(self.env.state_dim):
                for s2 in range(self.env.state_dim):
                    if self.transition_matrix_is_sparse:
                        if self.env.transition_matrix[a][s1, s2] > 0.0:
                            P.addToEntry(s1, s2, self.env.transition_matrix[a][s1, s2])
                    else:
                        if self.env.transition_matrix[a, s1, s2] > 0.0:
                            P.addToEntry(s1, s2, self.env.transition_matrix[a][s1, s2])

            # self.mdp.addMatrix(a, P)
            self.transitions_list.append(P)
            P = None

    def run(self):
        self.state_space = marmoteInterval(0, self.env.state_dim - 1)
        self.action_space = marmoteInterval(0, self.env.action_dim - 1)
        self.S = self.state_space.cardinal()
        self.A = self.action_space.cardinal()

        self._build_reward_matrix()

        self._build_transition_matrix()

        self.mdp = discountedMDP(
            "max",
            self.state_space,
            self.action_space,
            self.transitions_list,
            self.reward_matrix,
            self.discount,
        )

        self.start_time = time()

        self.opt: solutionMDP = self.mdp.valueIterationGS(self.epsilon, self.max_iter)

        self.runtime = time() - self.start_time

        self.value = np.array(
            [self.opt.getValueIndex(ss) for ss in range(self.env.state_dim)]
        )
        self.policy = None
