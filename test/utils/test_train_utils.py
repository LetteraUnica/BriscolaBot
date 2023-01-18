import unittest

from src.agents.RandomAgent import RandomAgent
from src.envs.two_player_briscola.BriscolaConstants import Constants
from src.envs.two_player_briscola.TwoPlayerBriscola import TwoPlayerBriscola
from src.utils.training_utils import play_all_moves_of_player, play_all_moves_of_players
from src.vectorizers.VectorizedEnv import VectorizedEnv


class TestTrainUtils(unittest.TestCase):
    def test_play_all_moves_of_player(self):
        vec_env = VectorizedEnv(lambda: TwoPlayerBriscola(), 128)
        n_actions = vec_env.single_action_space().n
        player_policy = RandomAgent(n_actions)
        vec_env.reset()
        for _ in range(Constants.deck_cards // (2 * Constants.n_agents) + 1):
            play_all_moves_of_player(vec_env.get_envs(), player_policy, "player_0")
            [self.assertEqual(agent, "player_1") for agent in vec_env.agent_selections()]
            play_all_moves_of_player(vec_env.get_envs(), player_policy, "player_1")
            [self.assertEqual(agent, "player_0") for agent in vec_env.agent_selections()]

    def test_play_all_moves_of_players(self):
        vec_env = VectorizedEnv(lambda: TwoPlayerBriscola(), 271)
        n_actions = vec_env.single_action_space().n
        player_policies = [RandomAgent(n_actions)] * 19
        vec_env.reset()
        for _ in range(Constants.deck_cards // (2 * Constants.n_agents) + 1):
            play_all_moves_of_players(vec_env.get_envs(), player_policies, "player_0")
            [self.assertEqual(agent, "player_1") for agent in vec_env.agent_selections()]
            play_all_moves_of_players(vec_env.get_envs(), player_policies, "player_1")
            [self.assertEqual(agent, "player_0") for agent in vec_env.agent_selections()]


if __name__ == '__main__':
    unittest.main()
