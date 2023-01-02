from typing import SupportsFloat, Any

from gymnasium import Env
from gymnasium.core import ObsType

from src.agents.Agent import Agent
from src.envs.two_player_briscola.TwoPlayerBriscola import TwoPlayerBriscola


class SinglePlayerBriscola(Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(self):
        self.briscola_env = TwoPlayerBriscola()
        self.adversary = None
        self.agent = self.briscola_env.agents[0]
        self.action_space = self.briscola_env.action_spaces[self.agent]
        self.observation_space = self.briscola_env.observation_spaces[self.agent]

    def set_adversary(self, adversary: Agent):
        self.adversary = adversary

    def reset(self, *, seed: int = None, options=None) -> tuple[ObsType, dict[str, Any]]:
        assert self.adversary is not None, "You must first add an adversary with the set_adversary method"

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
            observation = self.briscola_env.observe(self.briscola_env.agent_selection)
            self.briscola_env.step(self.adversary.get_action(observation))
            self._play_other_player_moves()
