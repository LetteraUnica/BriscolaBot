from typing import Any, Optional

from gymnasium import Space
from gymnasium.vector.utils import spaces
from pettingzoo import AECEnv

from src.env.two_player_briscola.BriscolaConstants import Constants
from src.env.two_player_briscola.BriscolaState import BriscolaState
from src.env.two_player_briscola.Card import Card


def card_priority(card: Card, hand_seed: int, briscola_seed: int):
    seed, rank = card.get_seed(), card.get_rank()

    if seed == briscola_seed:
        return Constants.cards_per_seed * card.get_priority() + 1

    if seed == hand_seed:
        return card.get_priority()

    return 0


class TwoPlayerBriscola(AECEnv):
    agents: list[int] = list(range(Constants.n_agents))
    num_agents: int = Constants.n_agents
    possible_agents: list[int] = agents.copy()
    max_num_agents: int = num_agents
    observation_spaces: dict[int, Space] = dict([(i, spaces.Dict({
        "seen_cards": spaces.MultiBinary(Constants.deck_cards),
        "briscola_card": spaces.Discrete(Constants.deck_cards + 1),
        "table_card": spaces.Discrete(Constants.deck_cards + 1),
        "hand_cards": spaces.MultiDiscrete([Constants.deck_cards] * Constants.hand_cards),
        "player_points": spaces.MultiDiscrete([Constants.total_points] * Constants.n_agents)
    })) for i in agents])
    action_spaces: dict[int, Space] = dict([(i, spaces.Discrete(Constants.hand_cards)) for i in agents])
    truncations: dict[int, int] = dict([(i, 0) for i in agents])

    def __init__(self):
        super().__init__()
        self.last_rewards = None
        self.state = BriscolaState()

        self.reset()

    @property
    def agent_selection(self) -> int:
        return self.state.get_current_player()

    @property
    def terminations(self) -> dict[int, bool]:
        return dict([(i, len(self.state.get_player_hand(i)) == 0) for i in self.agents])
    
    @property
    def reward(self) -> dict[int, int]:
        return self.last_rewards
    
    def seed(self, seed: Optional[int] = None):
        raise NotImplementedError()

    def observe(self, agent: Optional[int] = None) -> dict[str, Any]:
        return self.state.get_player_observation(agent)

    def render(self) -> str:
        return self.state.__repr__()

    def reset(self, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        self.last_rewards: dict[int, int] = {0: 0, 1: 0}
        self.state.reset()

    def close(self):
        pass

    def state(self):
        pass

    def step(self, card_thrown: int) -> None:
        assert card_thrown in self.action_spaces[self.agent_selection]

        if self.state.no_card_on_table():
            self.throw_card_on_table(card_thrown)
            self.state.next_player()
            self.last_rewards = {0: 0, 1: 1}

        second_thrown_card = self.state.pop_current_player_card(card_thrown)
        player_won, points = self.get_score(self.state.get_table_card(), second_thrown_card)
        self.state.add_points(points, player_won)

        if player_won == 1:
            self.state.reverse_player_order()

        self.end_turn()

        self.last_rewards[self.state.get_first_player()] = points

    @property
    def _cumulative_rewards(self):
        return self.state.player_points

    def is_over(self) -> bool:
        return self.state.no_card_in_player_hand()

    def throw_card_on_table(self, card_index: int):
        card_thrown = self.state.pop_current_player_card(card_index)
        self.state.set_table_card(card_thrown)

    def get_score(self, first_card: Card, second_card: Card) -> tuple[int, int]:
        hand_seed = first_card.get_seed()
        briscola_seed = self.state.get_briscola_seed()
        points = first_card.get_points() + second_card.get_points()
        if card_priority(first_card, hand_seed, briscola_seed) > card_priority(second_card, hand_seed, briscola_seed):
            return 0, points
        return 1, points

    def end_turn(self):
        self.state.next_player()
        self.state.deal_cards()
        self.state.reset_table_card()
