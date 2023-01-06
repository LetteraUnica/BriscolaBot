import unittest
from random import randint, choice, choices

import numpy as np

from src.envs.single_player_briscola.SinglePlayerBriscola import SinglePlayerBriscola
from src.envs.two_player_briscola.BriscolaConstants import Constants
from src.envs.two_player_briscola.TwoPlayerBriscola import TwoPlayerBriscola
from src.envs.two_player_briscola.utils import is_first_player_win, get_priority, get_seed


def seed_rank_to_card(seed: int, rank: int) -> int:
    return seed * Constants.cards_per_seed + rank


class MyTestCase(unittest.TestCase):
    def test_winner(self):
        # Ace wins against everything apart from briscola
        for card in range(1, Constants.cards_per_seed):
            self.assertTrue(is_first_player_win(0, card, 3))
            self.assertFalse(is_first_player_win(card, 0, 3))

        # Briscola wins against everything apart from higher briscolas
        for card in range(Constants.cards_per_seed):
            self.assertTrue(is_first_player_win(Constants.cards_per_seed + 1, card, 1))
            self.assertFalse(is_first_player_win(card, Constants.cards_per_seed + 1, 1))

        # Briscola Ace wins against everything
        for card in range(1, Constants.cards_per_seed):
            self.assertTrue(is_first_player_win(0, card, 0))
            self.assertFalse(is_first_player_win(card, 0, 0))

        # If second player throws a different seed card different from briscola it loses
        for card in range(Constants.cards_per_seed):
            self.assertTrue(is_first_player_win(Constants.cards_per_seed + 1, card, 1))
            self.assertFalse(is_first_player_win(card, Constants.cards_per_seed + 1, 1))

    def test_env_init(self):
        env = TwoPlayerBriscola()
        for _ in range(100):
            env.reset()
            self.assertEqual(len(env.game_state.deck), Constants.deck_cards - Constants.n_agents * Constants.hand_cards)
            self.assertEqual(env.game_state.deck[0], env.game_state.briscola_card)
            self.assertTrue(env.game_state.current_agent in env.agents)
            self.assertEqual(env.game_state.current_agent, env.agent_selection)
            self.assertEqual(len(env.game_state.seen_cards), 0)
            self.assertEqual(len(env.game_state.hand_cards), Constants.n_agents)
            for agent in env.game_state.hand_cards.keys():
                self.assertEqual(len(env.game_state.hand_cards[agent]), Constants.hand_cards)
                self.assertEqual(env.game_state.agent_points[agent], 0)
            self.assertEqual(env.game_state.table_card, Constants.null_card_number)
            self.assertEqual(env.game_state.num_moves, 0)

    def test_throw(self):
        env = TwoPlayerBriscola()
        for _ in range(100):
            env.reset()
            agent = env.game_state.current_agent
            action = randint(0, Constants.hand_cards - 1)
            thrown_card = env.game_state.hand_cards[agent][action]
            env.step(action)
            self.assertEqual(env.game_state.table_card, thrown_card)
            self.assertEqual(env.game_state.hand_cards[agent].count(thrown_card), 0)
            self.assertEqual(env.game_state.num_moves, 1)
            self.assertEqual(len(env.game_state.hand_cards[agent]), Constants.hand_cards - 1)

    def test_game_length(self):
        env = TwoPlayerBriscola()
        for _ in range(100):
            env.reset()
            num_moves = 0
            while not env.is_over():
                env.step(0)
                num_moves += 1
            self.assertEqual(num_moves, env.game_state.num_moves)
            self.assertEqual(env.game_state.num_moves, Constants.deck_cards)

    def test_game_end(self):
        env = TwoPlayerBriscola()
        for _ in range(100):
            env.reset()
            while not env.is_over():
                env.step(0)

            self.assertEqual(len(env.game_state.deck), 0)
            self.assertEqual(env.game_state.table_card, Constants.null_card_number)
            self.assertEqual(env.game_state.num_moves, Constants.deck_cards)
            for agent in env.game_state.hand_cards.keys():
                self.assertEqual(len(env.game_state.hand_cards[agent]), 0)

            self.assertEqual(sum(env.game_state.agent_points.values()), Constants.total_points)
            self.assertEqual(set(env.game_state.seen_cards), set(range(Constants.deck_cards)))

    def test_number_of_player_moves(self):
        env = TwoPlayerBriscola()
        for _ in range(100):
            env.reset()
            num_moves_per_player = {agent: 0 for agent in env.agents}
            while not env.is_over():
                num_moves_per_player[env.agent_selection] += 1
                env.step(0)
            for agent in env.agents:
                self.assertEqual(num_moves_per_player[agent], Constants.deck_cards // len(env.agents))

    def test_player_switch(self):
        env = TwoPlayerBriscola()
        for _ in range(100):
            env.reset()
            while not env.is_over():
                if env.game_state.table_card == Constants.null_card_number:
                    env.step(0)
                else:
                    first_card_played = env.game_state.table_card
                    second_card_played = env.game_state.hand_cards[env.agent_selection][0]

                    player = env.agent_selection
                    winner = is_first_player_win(first_card_played,
                                                 second_card_played,
                                                 get_seed(env.game_state.briscola_card))
                    env.step(0)
                    if winner:
                        self.assertEqual(env.agent_selection, env.other_player(player))
                    else:
                        self.assertEqual(env.agent_selection, player)

    def test_first_player_picks_first(self):
        env = TwoPlayerBriscola()
        for _ in range(100):
            env.reset()
            while not env.is_over() and len(env.game_state.deck) >= 2:
                first_card, second_card = env.game_state.deck[-1], env.game_state.deck[-2]
                env.step(0)
                env.step(0)
                self.assertTrue(first_card in env.game_state.hand_cards[env.agent_selection])
                self.assertTrue(second_card in env.game_state.hand_cards[env.other_player(env.agent_selection)])

    def test_observation_size(self):
        env = TwoPlayerBriscola()
        for _ in range(100):
            env.reset()
            while not env.is_over():
                observation_with_mask = env.observe(env.agent_selection)
                observation = observation_with_mask["observation"]
                self.assertEqual(observation.size, Constants.deck_cards * 4 + Constants.n_agents)
                self.assertTrue(observation_with_mask in env.observation_space(env.agent_selection))
                env.step(0)

    def test_reward(self):
        env = TwoPlayerBriscola()
        for _ in range(100):
            env.reset()
            prev_agent_points = env.game_state.agent_points.copy()
            for i in range(Constants.deck_cards):
                point_difference = env.game_state.agent_points[env.agent_selection] - prev_agent_points[
                    env.agent_selection]
                prev_agent_points[env.agent_selection] = env.game_state.agent_points[env.agent_selection]
                assert env.last()[1] * Constants.total_points == point_difference
                env.step(0)


if __name__ == '__main__':
    unittest.main()
