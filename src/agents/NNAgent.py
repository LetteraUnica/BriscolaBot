import numpy as np
from torch import nn, tensor
from torch.distributions import Categorical

from src.agents.Agent import Agent


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class NNAgent(nn.Module, Agent):
    def __init__(self, observation_shape, action_size, hidden_size=256):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(observation_shape).prod(), hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(observation_shape).prod(), hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, action_size), std=0.01),
        )

    def get_value(self, observations):
        return self.critic(observations)

    def get_actions(self, observations: tensor, action_masks: tensor = None):
        logits = self.actor(observations)
        if action_masks is not None:
            logits[~action_masks.bool()] = -1e8
        probs = Categorical(logits=logits)
        return probs.sample()

    def get_action_and_value(self, observations: tensor, action_masks: tensor = None, action=None):
        logits = self.actor(observations)
        if action_masks is not None:
            logits[~action_masks.bool()] = -1e8
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(observations)
