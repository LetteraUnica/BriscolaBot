from random import randint

from src.agents.Agent import Agent


class RandomAgent(Agent):
    def __init__(self, n_actions):
        self.n_actions = n_actions

    def get_action(self, observation):
        return randint(0, self.n_actions - 1)
