from abc import ABC, abstractmethod

from torch import tensor


class Agent(ABC):
    @abstractmethod
    def get_actions(self, observations: tensor, action_masks: tensor = None) -> tensor:
        pass
