from random import randint, choices

import numpy as np

from src.agents.Agent import Agent


class RandomAgent(Agent):
    def __init__(self, n_actions):
        self.n_actions = n_actions

    def get_action(self, observations, action_masks=None):
        if action_masks is None:
            return choices(range(self.n_actions), k=observations.shape[0])
        return [choices(range(self.n_actions), weights=action_mask, k=1)[0] for action_mask in action_masks]
