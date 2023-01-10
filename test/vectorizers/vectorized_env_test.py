import unittest

from src.envs.two_player_briscola.TwoPlayerBriscola import TwoPlayerBriscola
from src.vectorizers.VectorizedEnv import VectorizedEnv


class TestVecEnv(unittest.TestCase):
    def test_reset(self):
        vec_env = VectorizedEnv(lambda: TwoPlayerBriscola(), 128)
        vec_env.reset()
        game_states = [vec_env[i].game_state for i in range(len(vec_env))]
        vec_env.reset()
        [self.assertNotEqual(game_state, env.game_state) for game_state, env in zip(game_states, vec_env)]


if __name__ == '__main__':
    unittest.main()
