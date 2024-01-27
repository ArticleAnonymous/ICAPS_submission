import math as mt
from utils.generic_model import GenericModel
import numpy as np
from scipy.sparse import dok_matrix
from scipy.stats import binom, poisson

# From "Scheduling Services in a Queuing System with Impatience and Setup Costs", Emmanuel Hyon; Alain Jean-Marie

param_list = [
    {
        "proba_person_stay": 0.9,
        "arrival_poisson_rate": 0.9,
        "batch_cost": 1.5,
        "loss_cost": 2.0,
        "holding_cost": 0.0,
        "nb_serv": action_dim - 1,  # action space dimension
        "max_queue_length": state_dim - 1,  # state space dimension
    }
    for action_dim in [2, 10, 100]
    for state_dim in list(range(10, 100, 20))
    + list(range(100, 1100, 200))
    + list(range(1100, 10000, 1000))
]
default_params = param_list[0]


def probability_surv(current_number, surviving_people, proba_to_stay):
    return binom.pmf(surviving_people, current_number, proba_to_stay)


def probability_arrival(
    arrivals: int,
    current_number: int,
    arrival_poisson_rate: float,
    max_queue_length: int,
):
    assert 0 <= arrivals <= max_queue_length - current_number
    if arrivals < max_queue_length - current_number:
        return poisson.pmf(arrivals, arrival_poisson_rate)
    else:
        return 1 - sum(
            poisson.pmf(arrivals, arrival_poisson_rate)
            for arrivals in range(max_queue_length - current_number)
        )


def probability_trajectory(
    current_number,
    surviving_people,
    arrival_number,
    proba_to_stay,
    arrival_poisson_rate,
    max_queue_length,
):
    return probability_surv(
        current_number, surviving_people, proba_to_stay
    ) * probability_arrival(
        arrival_number, surviving_people, arrival_poisson_rate, max_queue_length
    )


def transition_function(
    yy: int,
    qq: int,
    zz: int,
    lambda_factor: float,
    max_queue_length: int,
    proba_to_stay: float,
):
    current_number_of_person = max(0, yy - qq)
    return sum(
        probability_trajectory(
            current_number_of_person,
            surviving_people,
            zz - surviving_people,
            proba_to_stay,
            lambda_factor,
            max_queue_length,
        )
        for surviving_people in range(min(zz, current_number_of_person) + 1)
    )


class Model(GenericModel):
    def __init__(self, params: dict = default_params):
        self.proba_person_stay = params["proba_person_stay"]
        assert 0 < self.proba_person_stay < 1, "Choose 0 < alpha < 1."

        self.arrival_poisson_rate = params["arrival_poisson_rate"]
        assert self.arrival_poisson_rate > 0, "Lambda should be positive."

        self.nb_serv = params["nb_serv"]
        assert self.nb_serv > 0, "Max batching capacity should be greater than 1."

        self.max_queue_length = params["max_queue_length"]
        assert self.max_queue_length > 0, "Queue should longer than 1."

        self.batch_cost = params["batch_cost"]
        self.loss_cost = params["loss_cost"]
        self.holding_cost = params["holding_cost"]
        self.q_cost = self.proba_person_stay * self.loss_cost + self.holding_cost

        self.state_dim = (
            self.max_queue_length + 1
        )  # On peut avoir 0 personnes ou max personnes dans la queue
        self.action_dim = (
            self.nb_serv + 1
        )  # On peut accueillir 0 personnes ou nb_serv personnes
        self.name = "{}_{}_queue_system".format(self.state_dim, self.action_dim)

    def _build_model(self):
        self.reward_matrix = np.zeros((self.state_dim, self.action_dim))
        self.transition_matrix: list = [
            dok_matrix((self.state_dim, self.state_dim), dtype=float)
            for _ in range(self.action_dim)
        ]

        for yy in range(self.state_dim):
            for qq in range(self.action_dim):
                self.reward_matrix[yy, qq] = qq * self.batch_cost + self.q_cost * max(
                    0, yy - qq
                )

                for zz in range(self.state_dim):
                    self.transition_matrix[qq][yy, zz] = transition_function(
                        yy,
                        qq,
                        zz,
                        self.arrival_poisson_rate,
                        self.max_queue_length,
                        self.proba_person_stay,
                    )

        self.transition_matrix = [matrix.tocsr() for matrix in self.transition_matrix]
