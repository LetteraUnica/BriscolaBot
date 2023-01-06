import torch
from torch.distributions import Categorical

from src.agents.Agent import Agent


class RandomAgent(Agent):
    def __init__(self, n_actions):
        self.n_actions = n_actions

    def get_actions(self, observations, action_masks=None):
        if action_masks is None:
            logits = torch.zeros((observations.shape[0], self.n_actions))
        else:
            logits = action_masks * 1e8

        return Categorical(logits=logits).sample()
