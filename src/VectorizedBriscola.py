from typing import Optional, Any

from src.env.briscola.TwoPlayerBriscola import TwoPlayerBriscola


class VectorizedBriscola:
    def __init__(self, n_envs=512):
        self.envs: list[TwoPlayerBriscola] = [TwoPlayerBriscola() for _ in range(n_envs)]

    def step(self, actions: list[int]) -> list[tuple[dict[str, Any], int, int, bool]]:
        return [env.step(action) for env, action in zip(self.envs, actions)]

    def reset(self, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None) -> None:
        [env.reset(seed, return_info, options) for env in self.envs]

    def seed(self, seed: Optional[int] = None) -> None:
        [env.seed(seed) for env in self.envs]

    def observe(self, agent: str) -> list[dict[str, Any]]:
        return [env.observe(agent) for env in self.envs]
