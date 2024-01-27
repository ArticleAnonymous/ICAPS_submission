"""
Solve a Discounted MDP with Linear Programming (dual problem) using Gurobi Linear Solver.
"""

from gurobipy import LinExpr, GRB, Model
import numpy as np
from utils.generic_model import GenericModel


def policy_bellman_value_to_value(
    env: GenericModel, value_function: np.ndarray, policy: np.ndarray, discount: float
) -> np.ndarray:
    """
    This function applies T^pi to the given value function.
    """
    assert policy.shape == (env.state_dim, env.action_dim)
    chosen_actions: np.ndarray = np.argmax(policy, axis=1)
    assert chosen_actions.shape == (env.state_dim,), "{}".format(chosen_actions.shape)

    q_value = np.array(
        [
            env.reward_matrix[:, aa]
            + discount * (env.transition_matrix[aa] @ value_function)
            for aa in range(env.action_dim)
        ]
    )

    return q_value.max(axis=0)


def policy_evaluation(
    env: GenericModel, policy: np.ndarray, discount: float, epsi: float = 1e-2
):
    """
    Returns V^pi
    """
    assert policy.shape == (
        env.state_dim,
        env.action_dim,
    ), "Policy shape is {} instead of {}".format(
        policy.shape, (env.state_dim, env.action_dim)
    )
    value = np.zeros((env.state_dim,))

    while True:
        value_old = value.copy()
        value = policy_bellman_value_to_value(env, value, policy, discount)
        assert value.shape == (
            env.state_dim,
        ), "Value shape is {} instead of {}".format(value.shape, (env.state_dim,))

        if np.linalg.norm(value - value_old, ord=np.inf) < epsi:
            return value


def gurobi_model_creation() -> Model:
    model = Model("MDP")
    model.setParam("OutputFlag", 0)
    model.setParam(GRB.Param.Threads, 1)
    model.setParam("LogToConsole", 0)
    return model


class Solver:
    def __init__(self, env: GenericModel, discount: float):
        self.env = env
        self.discount = discount
        self.name = "gurobi_discounted_dual"
        self.transition_matrix_is_sparse: bool = not isinstance(
            self.env.transition_matrix, np.ndarray
        )

    def _create_variables(self):
        # Dual variables
        self.var = {}
        for s in range(self.env.state_dim):
            for a in range(self.env.action_dim):
                self.var[(s, a)] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0.0)
        self.model.update()

    def _define_objective(self):
        self.obj = LinExpr()
        for s in range(self.env.state_dim):
            for a in range(self.env.action_dim):
                self.obj += self.env.reward_matrix[s, a] * self.var[(s, a)]
        self.model.setObjective(self.obj, GRB.MAXIMIZE)

    def _set_constraints(self):
        for s in range(self.env.state_dim):
            sum_value_over_action = sum(
                self.var[(s, a)] for a in range(self.env.action_dim)
            )
            sum_value_neighbors = sum(
                self.env.transition_matrix[aa][sp, s] * self.var[(sp, aa)]
                for aa in range(self.env.action_dim)
                for sp in range(self.env.state_dim)
            )
            self.model.addConstr(
                (sum_value_over_action - self.discount * sum_value_neighbors - 1 == 0),
                "Contrainte",
            )

    def run(self):
        self.model = gurobi_model_creation()

        # Primal variables
        self._create_variables()

        # Objective definition
        self._define_objective()

        # Constraints
        self._set_constraints()

        # Solving
        self.model.optimize()
        self.runtime = self.model.Runtime

        self.policy = np.zeros((self.env.state_dim, self.env.action_dim))
        for s in range(self.env.state_dim):
            for a in range(self.env.action_dim):
                self.policy[s, a] = self.var[(s, a)].X
        epsi = 1e-6
        self.value = np.zeros((self.env.state_dim))
        self.value = policy_evaluation(self.env, self.policy, self.discount, epsi)
