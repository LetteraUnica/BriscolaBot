from typing import Any, Optional

import numpy as np
from pettingzoo import AECEnv

from src.env.briscola.BriscolaConstants import Constants
from src.env.briscola.BriscolaState import BriscolaState
from src.env.briscola.Card import Card


def priority(card: Card, hand_seed: int, briscola_seed: int):
    seed, rank = card.get_seed(), card.get_rank()

    if seed == briscola_seed:
        return 10 * card.get_priority() + 1

    if seed == hand_seed:
        return card.get_priority()

    return 0


class TwoPlayerBriscola(AECEnv):
    def seed(self, seed: Optional[int] = None):
        raise NotImplementedError()

    def observe(self, agent: str) -> dict[str, Any]:
        return self.state.get_current_player_observation()

    def render(self) -> str:
        return self.state.__repr__()

    def state(self) -> np.ndarray:
        raise NotImplementedError()

    def __init__(self):
        super().__init__()

        self.state = BriscolaState()

        self.reset()

    def reset(self, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None) -> \
            dict[str, Any]:
        self.state.reset()
        return self.state.get_current_player_observation()

    def is_over(self) -> bool:
        return self.state.no_card_in_player_hand()

    def step(self, card_index: int) -> tuple[dict[str, Any], int, int, bool]:
        assert 0 <= card_index < Constants.hand_cards

        if self.state.no_card_on_table():
            self.throw_card_on_table(card_index)
            self.state.next_player()
            return self.state.get_current_player_observation(), -1, -1, self.is_over()

        second_thrown_card = self.state.pop_current_player_card(card_index)
        player_won, points = self.get_score(self.state.get_table_card(), second_thrown_card)
        self.state.add_points(points, player_won)

        if player_won == 1:
            self.state.reverse_player_order()

        self.end_turn()

        return self.state.get_current_player_observation(), self.state.get_first_player(), points, self.is_over()

    def throw_card_on_table(self, card_index: int):
        card_thrown = self.state.pop_current_player_card(card_index)
        self.state.set_table_card(card_thrown)

    def get_score(self, first_card: Card, second_card: Card) -> tuple[int, int]:
        hand_seed = first_card.get_seed()
        briscola_seed = self.state.get_briscola_seed()
        points = first_card.get_points() + second_card.get_points()
        if priority(first_card, hand_seed, briscola_seed) > priority(second_card, hand_seed, briscola_seed):
            return 0, points
        return 1, points

    def end_turn(self):
        self.state.next_player()
        self.state.deal_cards()
        self.state.reset_table_card()
