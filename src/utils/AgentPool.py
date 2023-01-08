import numpy as np
from numpy import ndarray
from scipy.special import softmax

from src.agents import Agent


class AgentPool:
    def __init__(self, k: int = 16):
        self.k: int = k
        self.ratings: list[float] = []
        self.agents: list[Agent] = []
        self.sigmoid = lambda x: 1 / (1 + 10**(-x))

    def sample_agents(self, n_agents: int = 1) -> tuple[list[Agent], ndarray]:
        p = softmax(self.ratings)
        indexes = np.random.choice(len(self.agents), size=n_agents, p=p)
        return [self.agents[index] for index in indexes], indexes

    def update_ratings(self, opponent_rating: float, actual_scores: ndarray, agent_indexes: ndarray):
        my_ratings = np.array(self.ratings)[agent_indexes]
        excepted_scores = self.sigmoid(my_ratings - opponent_rating)
        my_ratings = my_ratings + self.k*(actual_scores - excepted_scores)
        for rating, index in zip(my_ratings, agent_indexes):
            self.ratings[index] = rating

    def add_agent(self, agent: Agent, rating: float):
        self.agents.append(agent)
        self.ratings.append(rating)

    def clean_pool(self, max_length: int):
        min_index = len(self.agents) - max_length
        self.ratings = self.ratings[min_index:]
        self.agents = self.agents[min_index:]