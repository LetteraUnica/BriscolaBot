import random
from dataclasses import dataclass
from functools import cache
from random import sample
from typing import Optional, Any

from gymnasium import Space
from gymnasium.spaces import Dict, MultiBinary, Discrete, MultiDiscrete
from pettingzoo import AECEnv

from src.briscola_env.two_player_briscola.BriscolaConstants import Constants
from src.briscola_env.two_player_briscola.utils import get_seed, get_points, get_priority


@dataclass
class State:
    deck: list[int]
    seen_cards: list[int]
    hand_cards: dict[str, list[int]]
    table_card: int
    briscola_card: int
    current_agent: str
    agent_points: dict[str, float]

    def get_number_of_card_in_hand(self, agent: str) -> int:
        return len(self.hand_cards[agent])

    def extract_cards(self, n_cards: int) -> list[int]:
        cards_to_extract = min(len(self.deck), n_cards)
        return [self.deck.pop() for _ in range(cards_to_extract)]

    def add_cards_to(self, agent_id: str, cards: list[int]):
        self.hand_cards[agent_id].extend(cards)

    def pop_card_of_agent(self, agent: str, card_index: int) -> int:
        return self.hand_cards[agent].pop(card_index)

    def add_seen_cards(self, cards: list[int]):
        self.seen_cards.extend(cards)


def create_deck():
    return sample(range(Constants.deck_cards), k=Constants.deck_cards)


class Briscola(AECEnv):
    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        self.state = None
        self.possible_agents: list[str] = ["player_" + str(agent) for agent in range(Constants.n_agents)]
        self.agents = self.possible_agents.copy()
        self.rewards: dict[str, float] = {agent: 0 for agent in self.agents}
        self.reset(seed)

    @cache
    def observation_space(self, agent: str) -> Space:
        return Dict({
            "seen_cards": MultiBinary(Constants.deck_cards),
            "briscola_card": Discrete(Constants.deck_cards),
            "table_card": Discrete(Constants.deck_cards + 1),
            "hand_cards": MultiDiscrete([Constants.deck_cards + 1] * Constants.hand_cards),
            "agent_points": MultiDiscrete([Constants.total_points + 1] * Constants.n_agents)
        })

    def reset(self, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None) -> None:
        self.seed(seed)
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}
        self.num_moves = 0
        deck = create_deck()

        self.state = State(deck=deck,
                           seen_cards=[],
                           hand_cards=dict([(agent, []) for agent in self.agents]),
                           table_card=Constants.null_card_number,
                           briscola_card=deck[0],
                           current_agent=sample(self.agents, k=1)[0],
                           agent_points=dict([(agent, 0) for agent in self.agents]))

        self.zero_out_reward()
        self.deal_cards(Constants.hand_cards)

    @cache
    def action_space(self, agent: str) -> Space:
        return Discrete(len(self.state.hand_cards[agent]))

#    @property
#    def action_spaces(self) -> dict[str, Space]:
#        return dict([(agent, self.action_space(agent)) for agent in self.agents])

    def zero_out_reward(self):
        [self.rewards.update({agent: 0}) for agent in self.agents]

    @property
    def agent_selection(self) -> str:
        return self.state.current_agent

    @property
    def terminations(self) -> dict[str, bool]:
        return dict([(agent, self.state.get_number_of_card_in_hand(agent) == 0) for agent in self.agents])

    @property
    def truncations(self) -> dict[str, bool]:
        return self.terminations

    def step(self, action: int) -> None:
        assert action in self.action_space(self.agent_selection)

        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            return

        self._cumulative_rewards[self.agent_selection] = 0

        if self.state.table_card == Constants.null_card_number:
            self.state.table_card = self.state.pop_card_of_agent(self.agent_selection, action)
            self.zero_out_reward()
        else:
            first_card = self.state.table_card
            second_card = self.state.pop_card_of_agent(self.agent_selection, action)
            total_points = get_points(first_card) + get_points(second_card)
            hand_seed, briscola_seed = get_seed(first_card), get_seed(self.state.briscola_card)

            if get_priority(first_card, hand_seed, briscola_seed) > get_priority(second_card, hand_seed, briscola_seed):
                winner = self.other_player(self.agent_selection)
            else:
                winner = self.agent_selection
                self.invert_player_turn()

            self.rewards[winner] = total_points
            self.state.agent_points[winner] += total_points
            self.state.add_seen_cards([first_card, second_card])
            self.state.table_card = Constants.null_card_number
            self.deal_cards(1)

        self._accumulate_rewards()
        self.next_turn()

    def other_player(self, agent: str) -> str:
        return self.agents[0] if agent == self.agents[1] else self.agents[1]

    def next_turn(self):
        self.num_moves += 1
        self.state.current_agent = self.other_player(self.state.current_agent)

    def invert_player_turn(self):
        self.next_turn()

    def seed(self, random_seed: Optional[int] = None) -> None:
        random.seed(random_seed)

    def observe(self, agent: str) -> dict[str, Any]:
        number_null_cards = Constants.hand_cards - len(self.state.hand_cards[agent])
        return {
            "seen_cards": [1 if card in self.state.seen_cards else 0 for card in range(Constants.deck_cards)],
            "briscola_card": self.state.briscola_card,
            "table_card": self.state.table_card,
            "hand_cards": self.state.hand_cards[agent] + [Constants.null_card_number] * number_null_cards,
            "agent_points": [self.state.agent_points[agent], self.state.agent_points[self.other_player(agent)]]
        }

    def render(self) -> str:
        return self.state.__repr__()

    def state(self) -> State:
        return self.state

    def deal_cards(self, n_cards: int):
        for agent in self.agents:
            cards = self.state.extract_cards(n_cards)
            self.state.add_cards_to(agent, cards)

    def close(self):
        super().close()
