"""
Solve a Discounted MDP with Linear Programming (primal problem) using Gurobi Linear Solver.
"""

from gurobipy import LinExpr, GRB
import numpy as np
from utils.generic_model import GenericModel
from gurobipy import Model, GRB


def gurobi_model_creation() -> Model:
    model = Model("MDP")
    model.setParam("OutputFlag", 0)
    # model.setParam(GRB.Param.Threads, 1)
    model.setParam("LogToConsole", 0)
    return model


class Solver:
    def __init__(self, env: GenericModel, discount: float):
        self.env = env
        self.discount = discount
        self.name = "gurobi_discounted_primal"

    def _create_variables(self):
        self.var = {}
        for ss in range(self.env.state_dim):
            self.var[ss] = self.model.addVar(
                vtype=GRB.CONTINUOUS, name="v({})".format(ss), lb=-np.inf, ub=np.inf
            )

    def _define_objective(self):
        self.model.setObjective(
            sum(self.var[s] for s in range(self.env.state_dim)), GRB.MINIMIZE
        )
        self.model.update()

    def _set_constraints(self):
        for ss in range(self.env.state_dim):
            for aa in range(self.env.action_dim):
                neighors_value_sum = sum(
                    self.env.transition_matrix[aa][ss, ss2] * self.var[ss2]
                    for ss2 in range(self.env.state_dim)
                )
                self.model.addConstr(
                    self.var[ss]
                    >= self.env.reward_matrix[ss, aa]
                    + self.discount * neighors_value_sum
                )
        self.model.update()

    def run(self):
        self.model: Model = gurobi_model_creation()

        # Variables
        self._create_variables()

        # Objective
        self._define_objective()

        # Constraints
        self._set_constraints()

        self.model.optimize()
        self.runtime = self.model.Runtime

        self.value = np.array(self.model.x)
