from dataclasses import dataclass
from typing import Optional, Any, Union

import numpy as np
from gymnasium import Space
from gymnasium.spaces import Dict, Discrete, Box
from numpy.random import Generator
from pettingzoo import AECEnv

from src.envs.two_player_briscola.BriscolaConstants import Constants
from src.envs.two_player_briscola.utils import get_seed, get_points, get_priority


@dataclass
class State:
    deck: list[int]
    seen_cards: list[int]
    hand_cards: dict[str, list[int]]
    table_card: int
    briscola_card: int
    current_agent: str
    agent_points: dict[str, float]
    num_moves: int

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

    def number_of_agent_cards(self, agent: str) -> int:
        return len(self.hand_cards[agent])


class TwoPlayerBriscola(AECEnv):
    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        self.rng: Optional[Generator] = None
        self.game_state: Union[State, None] = None

        self.possible_agents: list[str] = ["player_" + str(agent) for agent in range(Constants.n_agents)]
        self.agents = self.possible_agents.copy()

        self.action_spaces = {agent: Discrete(Constants.hand_cards) for agent in self.agents}
        self.observation_spaces = {
            agent: Dict(
                {
                    "observation": Box(low=0, high=1, shape=(Constants.deck_cards * 4 + Constants.n_agents,),
                                       dtype=np.float32),
                    "action_mask": Box(low=0, high=1, shape=(Constants.hand_cards,), dtype=np.float32),
                }) for agent in self.agents
        }

        self.infos = {i: {} for i in self.agents}

        self.reset(seed)

    def reset(self, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None) -> None:
        self.seed(seed)

        self.agents = self.possible_agents.copy()
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.truncations = {i: False for i in self.agents}

        deck = np.arange(Constants.deck_cards)
        self.rng.shuffle(deck)
        self.game_state = State(deck=list(deck),
                                seen_cards=[],
                                hand_cards=dict([(agent, []) for agent in self.agents]),
                                table_card=Constants.null_card_number,
                                briscola_card=deck[0],
                                current_agent=self.rng.choice(self.agents, size=1)[0],
                                agent_points=dict([(agent, 0) for agent in self.agents]),
                                num_moves=0)

        self.deal_cards(Constants.hand_cards)

    def seed(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)

    def observation_space(self, agent: str) -> Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> Space:
        return self.action_spaces[agent]

    def zero_out_reward(self):
        [self.rewards.update({agent: 0}) for agent in self.agents]

    @property
    def agent_selection(self) -> str:
        return self.game_state.current_agent

    @property
    def terminations(self) -> dict[str, bool]:
        return dict([(agent, self.game_state.get_number_of_card_in_hand(agent) == 0) for agent in self.agents])

    def observe(self, agent: str) -> dict[str, Any]:
        observation = np.zeros((Constants.deck_cards, 4), dtype=np.float32)
        observation[self.game_state.seen_cards, 0] = 1
        observation[self.game_state.briscola_card, 1] = 1
        if self.game_state.table_card < Constants.null_card_number:
            observation[self.game_state.table_card, 2] = 1
        observation[self.game_state.hand_cards[agent], 3] = 1
        observation = np.concatenate((
            observation.flatten(),
            np.array([
                self.game_state.agent_points[agent] / Constants.total_points,
                self.game_state.agent_points[self.other_player(agent)] / Constants.total_points
            ])
        ), dtype=np.float32)

        action_mask = np.zeros((Constants.hand_cards,), dtype=np.int8)
        action_mask[self.legal_actions(agent)] = 1

        return {"observation": observation, "action_mask": action_mask}

    def legal_actions(self, agent: str) -> list[int]:
        return list(range(self.game_state.number_of_agent_cards(agent)))

    def step(self, action: int) -> None:
        assert not self.terminations[self.agent_selection] or not self.truncations, "game finished"

        assert action in self.legal_actions(self.agent_selection), "invalid action"

        if self.game_state.table_card == Constants.null_card_number:
            self.game_state.table_card = self.game_state.pop_card_of_agent(self.agent_selection, action)
            self.zero_out_reward()
        else:
            first_card = self.game_state.table_card
            second_card = self.game_state.pop_card_of_agent(self.agent_selection, action)
            total_points = get_points(first_card) + get_points(second_card)
            hand_seed, briscola_seed = get_seed(first_card), get_seed(self.game_state.briscola_card)

            if get_priority(first_card, hand_seed, briscola_seed) > get_priority(second_card, hand_seed, briscola_seed):
                winner = self.other_player(self.agent_selection)
            else:
                winner = self.agent_selection
                self.invert_player_turn()

            self.rewards[winner] = total_points
            self.game_state.agent_points[winner] += total_points
            self.game_state.add_seen_cards([first_card, second_card])
            self.game_state.table_card = Constants.null_card_number
            self.deal_cards(1)

        self._cumulative_rewards[self.agent_selection] = 0
        self._accumulate_rewards()
        self.next_turn()

    def other_player(self, agent: str) -> str:
        return self.agents[0] if agent == self.agents[1] else self.agents[1]

    def next_turn(self):
        self.game_state.num_moves += 1
        self.game_state.current_agent = self.other_player(self.game_state.current_agent)

    def invert_player_turn(self):
        self.next_turn()

    def render(self) -> str:
        return self.game_state.__repr__()

    def state(self) -> State:
        pass

    def deal_cards(self, n_cards: int):
        for agent in self.agents:
            cards = self.game_state.extract_cards(n_cards)
            self.game_state.add_cards_to(agent, cards)

    def close(self):
        super().close()
