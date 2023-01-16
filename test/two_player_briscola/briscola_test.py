import unittest
from random import choice

from src.envs.two_player_briscola.BriscolaConstants import Constants
from src.envs.two_player_briscola.TwoPlayerBriscola import TwoPlayerBriscola
from src.envs.two_player_briscola.utils import is_first_player_win, get_seed


def seed_rank_to_card(seed: int, rank: int) -> int:
    return seed * Constants.cards_per_seed + rank


def get_random_action(env: TwoPlayerBriscola) -> int:
    return choice(env.legal_actions(env.agent_selection))


def play_random_actions(env: TwoPlayerBriscola, n_actions: int = 1):
    for _ in range(n_actions):
        env.step(get_random_action(env))


class TestBriscola(unittest.TestCase):

    def test_hand_winner(self):
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
            thrown_card = get_random_action(env)
            env.step(thrown_card)
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
                play_random_actions(env)
                num_moves += 1
            self.assertEqual(num_moves, env.game_state.num_moves)
            self.assertEqual(env.game_state.num_moves, Constants.deck_cards)

    def test_game_end(self):
        env = TwoPlayerBriscola()
        for _ in range(100):
            env.reset()
            while not env.is_over():
                play_random_actions(env)

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
                play_random_actions(env)
            for agent in env.agents:
                self.assertEqual(num_moves_per_player[agent], Constants.deck_cards // len(env.agents))

    def test_player_switch(self):
        env = TwoPlayerBriscola()
        for _ in range(100):
            env.reset()
            while not env.is_over():
                if env.game_state.table_card == Constants.null_card_number:
                    play_random_actions(env)
                else:
                    first_card_played = env.game_state.table_card
                    second_card_played = get_random_action(env)

                    player = env.agent_selection
                    winner = is_first_player_win(first_card_played,
                                                 second_card_played,
                                                 get_seed(env.game_state.briscola_card))
                    env.step(second_card_played)
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
                play_random_actions(env, n_actions=2)
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
                play_random_actions(env)

    def test_reward(self):
        env = TwoPlayerBriscola()
        for _ in range(100):
            env.reset()
            prev_agent_points = env.game_state.agent_points.copy()
            for i in range(Constants.deck_cards):
                point_difference = env.game_state.agent_points[env.agent_selection] - prev_agent_points[
                    env.agent_selection]
                prev_agent_points[env.agent_selection] = env.game_state.agent_points[env.agent_selection]
                self.assertAlmostEqual(env.last()[1] * Constants.total_points, point_difference)
                play_random_actions(env)

    def test_total_points(self):
        env = TwoPlayerBriscola()
        for _ in range(100):
            env.reset()
            while not env.is_over():
                play_random_actions(env)
            self.assertEqual(sum(env.game_state.agent_points.values()), Constants.total_points)

    def test_total_reward(self):
        env = TwoPlayerBriscola()
        for _ in range(100):
            env.reset()
            total_reward = 0.
            while not env.is_over():
                play_random_actions(env)
                total_reward += env.last()[1]
            self.assertAlmostEqual(total_reward, 1.)

    def test_is_over(self):
        env = TwoPlayerBriscola()
        for _ in range(100):
            env.reset()
            while not env.is_over():
                self.assertEqual(env.is_over(), env.game_state.num_moves == Constants.deck_cards)
                self.assertEqual(env.is_over(), len(env.game_state.seen_cards) == Constants.deck_cards)
                self.assertEqual(env.is_over(), sum(len(env.game_state.hand_cards[agent]) for agent in env.agents) == 0)
                self.assertEqual(env.is_over(), all(env.terminations.values()))
                play_random_actions(env)

            self.assertEqual(env.is_over(), sum(env.game_state.agent_points.values()) == Constants.total_points)
            self.assertEqual(env.is_over(), len(env.game_state.deck) == 0)
            self.assertEqual(env.is_over(), env.game_state.table_card == Constants.null_card_number)

    def test_is_even(self):
        env = TwoPlayerBriscola()
        for _ in range(100):
            env.reset()
            while not env.is_over():
                play_random_actions(env)
            self.assertEqual(env.is_even(), env.game_state.agent_points[env.agents[0]] == Constants.total_points // 2)

    def test_winner(self):
        env = TwoPlayerBriscola()
        for _ in range(100):
            env.reset()
            while not env.is_over():
                play_random_actions(env)

            winner = env.game_winner()
            if winner is None:
                self.assertTrue(env.is_even())
            else:
                self.assertTrue(env.game_state.agent_points[winner] > Constants.total_points // 2)
                self.assertTrue(env.game_state.agent_points[winner] >
                                env.game_state.agent_points[env.other_player(winner)])

    def test_game_outcome(self):
        env = TwoPlayerBriscola()
        for _ in range(100):
            env.reset()
            while not env.is_over():
                play_random_actions(env)

            self.assertAlmostEqual(sum(env.get_game_outcome(agent) for agent in env.agents), 1.)
            if env.is_even():
                [self.assertAlmostEqual(env.get_game_outcome(agent), 0.5) for agent in env.agents]
            else:
                self.assertAlmostEqual(env.get_game_outcome(env.game_winner()), 1.)
                self.assertAlmostEqual(env.get_game_outcome(env.other_player(env.game_winner())), 0.)


if __name__ == '__main__':
    unittest.main()
