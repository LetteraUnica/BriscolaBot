from typing import Callable, Any, Union

from gymnasium import Space
from numpy import ndarray
from pettingzoo import AECEnv

from src.envs.two_player_briscola import TwoPlayerBriscola


class VectorizedEnv:
    def __init__(self, env_fn: Callable[[], AECEnv], n_envs: int):
        self.envs: list[AECEnv] = [env_fn() for _ in range(n_envs)]

    def reset(self):
        [env.reset() for env in self.envs]

    def step(self, actions: ndarray, **kwargs):
        [env.step(action, **kwargs) for env, action in zip(self.envs, actions)]

    def observe(self, agent: str) -> list[dict[str, ndarray]]:
        return [env.observe(agent) for env in self.envs]

    def last(self) -> list[tuple[dict[str, ndarray], float, bool, bool, dict[str, Any]]]:
        return [env.last() for env in self.envs]

    def single_observation_space(self) -> Space:
        return self.envs[0].observation_space(self.envs[0].agent_selection)

    def single_action_space(self) -> Space:
        return self.envs[0].action_space(self.envs[0].agent_selection)

    def agent_selections(self) -> list[str]:
        return [env.agent_selection for env in self.envs]

    def get_envs(self) -> Union[list[AECEnv], list[TwoPlayerBriscola]]:
        return self.envs

    def __len__(self):
        return len(self.envs)

    def __getitem__(self, index):
        return self.envs[index]
