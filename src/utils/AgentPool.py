import numpy as np
from numpy import ndarray
from scipy.special import softmax, logit

from src.agents import Agent


class AgentPool:
    def __init__(self, max_size: int, nu: float = 0.1):
        self.max_size = max_size
        self.nu: float = nu
        self.ratings: list[float] = []
        self.agents: list[Agent] = []

    def get_sampling_probability(self) -> ndarray:
        return softmax(self.ratings)

    def sample_agents(self, n_agents: int = 1) -> tuple[list[Agent], ndarray]:
        p = self.get_sampling_probability()
        indexes = np.random.choice(len(self), size=min(n_agents, len(self)), p=p, replace=False)
        return [self.agents[index] for index in indexes], indexes

    def update_ratings(self, opponent_rating: float, actual_scores: ndarray, agent_indexes: ndarray) -> float:
        # my_ratings = np.array(self.ratings)[agent_indexes]
        # excepted_scores = expit(my_ratings - opponent_rating)
        # k = self.nu / (len(self) * self.get_sampling_probability()[agent_indexes])
        # my_ratings = my_ratings + k * (actual_scores - excepted_scores)
        my_ratings = logit(actual_scores)
        # opponent_rating = opponent_rating - (k * (actual_scores - excepted_scores)).sum()
        for index, rating in zip(agent_indexes, my_ratings):
            self.ratings[index] = rating

        return opponent_rating

    def add_agent(self, agent: Agent, rating: float = None):
        if rating is None:
            rating = 0.
        self.agents.append(agent)
        self.ratings.append(rating)
        if len(self) > self.max_size:
            self.clean_pool(self.max_size)

    def add_agents(self, agents: list[Agent], ratings: list[float]):
        assert len(agents) == len(ratings), "agents and ratings must have the same length"
        self.agents.extend(agents)
        self.ratings.extend(ratings)

    def clean_pool(self, max_length: int):
        min_index = len(self) - max_length
        self.ratings = self.ratings[min_index:]
        self.agents = self.agents[min_index:]

    def __len__(self):
        return len(self.agents)
