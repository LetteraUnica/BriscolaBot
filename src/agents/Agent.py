from abc import ABC, abstractmethod
from typing import Any

from numpy import ndarray
from torch import tensor


class Agent(ABC):
    @abstractmethod
    def get_action(self, observations: tensor, action_masks: tensor = None) -> Any:
        pass
