from torch import tensor, nn

from src.agents.Agent import Agent


class GreedyAgent(nn.Module, Agent):
    def __init__(self, policy: nn.Module, action_size: int, name: str = "Greedy-Agent"):
        super().__init__()
        self.actor = policy
        self.action_size = action_size
        self.name = name

    def get_name(self) -> str:
        return self.name

    def get_logits(self, observations: tensor, action_masks: tensor) -> tensor:
        logits = self.actor(observations)
        if action_masks is not None:
            logits[~action_masks.bool()] = -1e8
        return logits

    def get_actions(self, observations: tensor, action_masks: tensor = None):
        return self.get_logits(observations, action_masks).argmax(dim=1)

    def forward(self, inputs: tensor):
        observation, action_mask = inputs[:, :-self.action_size], inputs[:, -self.action_size:]
        return self.get_actions(observation, action_mask)
