from typing import Callable

from gymnasium import Space
from pettingzoo import AECEnv


class VectorizedEnv:
    def __init__(self, env_fn: Callable[[], AECEnv], n_envs: int):
        self.envs: list[AECEnv] = [env_fn() for _ in range(n_envs)]

    def reset(self):
        [env.reset() for env in self.envs]

    def step(self, actions: list[int]):
        [env.step(action) for env, action in zip(self.envs, actions)]

    def observe(self, agent: str):
        return [env.observe(agent) for env in self.envs]

    def last(self):
        return [env.last() for env in self.envs]

    def single_observation_space(self) -> Space:
        return self.envs[0].observation_space(self.envs[0].agent_selection)

    def single_action_space(self) -> Space:
        return self.envs[0].action_space(self.envs[0].agent_selection)

    def __len__(self):
        return len(self.envs)

    def __getitem__(self, index):
        return self.envs[index]
