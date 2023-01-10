import torch
from torch.distributions import Categorical

from src.agents.Agent import Agent


class RandomAgent(Agent):
    def __init__(self, n_actions):
        self.n_actions = n_actions

    def get_actions(self, observations, action_masks=None):
        weights = torch.ones((observations.shape[0], self.n_actions))
        if action_masks is not None:
            weights[~action_masks.bool()] = 0.

        return Categorical(probs=weights).sample()
