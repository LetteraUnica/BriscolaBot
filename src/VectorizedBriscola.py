from typing import Optional, Any

from pettingzoo import AECEnv

from src.env.briscola.TwoPlayerBriscola import TwoPlayerBriscola


class VectorizedBriscola:
    def __init__(self, n_envs=512):
        self.envs: list[AECEnv] = [TwoPlayerBriscola() for _ in range(n_envs)]

    def step(self, actions: list[int]) -> None:
        for env, action in zip(self.envs, actions):
            env.step(action)

    def reset(self, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None) -> None:
        for env in self.envs:
            env.reset(seed, return_info, options)

    def seed(self, seed: Optional[int] = None) -> None:
        for env in self.envs:
            env.seed(seed)

    def observe(self, agent: str) -> list[dict[str, Any]]:
        return [env.observe(agent) for env in self.envs]
