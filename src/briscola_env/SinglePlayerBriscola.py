from abc import abstractmethod, ABC
from typing import SupportsFloat, Any

import numpy as np
from gymnasium import Env
from gymnasium.core import ObsType

from src.briscola_env.two_player_briscola.Briscola import Briscola


class Agent(ABC):
    @abstractmethod
    def get_action(self, observation):
        pass


class SinglePlayerBriscola(Env):
    def __init__(self, other_player: Agent):
        self.briscola_env = Briscola()
        self.other_player = other_player
        self.agent = self.briscola_env.agents[0]

    def reset(self, *, seed: int = None, options=None) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.briscola_env.reset()
        self._play_other_player_moves()
        return self.briscola_env.observe(self.agent), self.briscola_env.infos[self.agent]

    def close(self):
        self.briscola_env.close()
        super().close()

    def step(self, action: int) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.briscola_env.step(action)
        self._play_other_player_moves()

        return (self.briscola_env.observe(self.agent),
                self.briscola_env.rewards[self.agent],
                self.briscola_env.terminations[self.agent],
                self.briscola_env.truncations[self.agent],
                self.briscola_env.infos[self.agent])

    def render(self) -> str:
        return self.briscola_env.render()

    def _play_other_player_moves(self):
        if self.briscola_env.agent_selection == self.briscola_env.agents[1]:
            self.briscola_env.step(self.other_player.get_action())
            self._play_other_player_moves()
