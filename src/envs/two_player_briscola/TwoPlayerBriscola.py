from dataclasses import dataclass
from typing import Optional, Union
from warnings import warn

import numpy as np
from gymnasium import Space
from gymnasium.spaces import Dict, Discrete, Box
from numpy.random import Generator
from pettingzoo import AECEnv

from src.envs.two_player_briscola.BriscolaConstants import Constants
from src.envs.two_player_briscola.utils import get_seed, get_rank, is_first_player_win, \
    get_cards_points


def card_to_string(card: int) -> str:
    if card == Constants.null_card_number:
        return "NA"
    seed, rank = get_seed(card), get_rank(card)
    return f"{rank}{Constants.seed_representation[seed]}"


def cards_to_string(cards: list[int]) -> str:
    return " ".join(card_to_string(card) for card in cards)


@dataclass
class State:
    deck: list[int]
    thrown_cards_player: list[str]
    thrown_cards: list[int]
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

    def pop_card_of_agent(self, agent: str, card: int) -> int:
        index = self.hand_cards[agent].index(card)
        return self.hand_cards[agent].pop(index)

    def add_seen_cards(self, cards: list[int], players: list[str]):
        self.thrown_cards.extend(cards)
        self.thrown_cards_player.extend(players)

    def number_of_agent_cards(self, agent: str) -> int:
        return len(self.hand_cards[agent])

    def __repr__(self):
        hand_cards = {k: cards_to_string(v) for k, v in self.hand_cards.items()}
        return f"deck: {cards_to_string(self.deck)}\n" \
               f"seen_cards: {cards_to_string(self.thrown_cards)}\n" \
               f"hand_cards: {hand_cards}\n" \
               f"table_card: {card_to_string(self.table_card)}\n" \
               f"briscola_card: {card_to_string(self.briscola_card)}\n" \
               f"current_agent: {self.current_agent}\n" \
               f"agent_points: {self.agent_points}\n" \
               f"num_moves: {self.num_moves}\n"


class TwoPlayerBriscola(AECEnv):
    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        self.rng: Optional[Generator] = None
        self.game_state: Union[State, None] = None

        self.possible_agents: list[str] = ["player_" + str(agent) for agent in range(Constants.n_agents)]
        self.agents = self.possible_agents.copy()

        self.action_spaces = {agent: Discrete(Constants.deck_cards) for agent in self.agents}
        self.observation_spaces = {
            agent: Dict(
                {
                    "observation": Box(low=0, high=1, shape=(4 * Constants.deck_cards + Constants.n_agents,),
                                       dtype=np.float32),
                    "action_mask": Box(low=0, high=1, shape=(Constants.deck_cards,), dtype=np.int64),
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
                                thrown_cards=[],
                                thrown_cards_player=[],
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

    def observe(self, agent: str) -> dict[str, np.ndarray]:
        observation = np.zeros((4, Constants.deck_cards), dtype=np.float32)
        observation[0, self.game_state.thrown_cards] = 1
        observation[1, self.game_state.briscola_card] = 1
        if self.game_state.table_card < Constants.null_card_number:
            observation[2, self.game_state.table_card] = 1
        observation[3, self.game_state.hand_cards[agent]] = 1
        observation = np.concatenate((
            observation.ravel(),
            np.array([
                self.game_state.agent_points[agent] / Constants.total_points,
                self.game_state.agent_points[self.other_player(agent)] / Constants.total_points
            ])
        ), dtype=np.float32)

        action_mask = np.zeros((Constants.deck_cards,), dtype=np.int64)
        action_mask[self.legal_actions(agent)] = 1

        return {"observation": observation, "action_mask": action_mask}

    def legal_actions(self, agent: str) -> list[int]:
        return self.game_state.hand_cards[agent]

    def step(self, action: int) -> None:
        assert not self.terminations[self.agent_selection] or not self.truncations, "game finished"

        if action not in self.legal_actions(self.agent_selection):
            action = self.legal_actions(self.agent_selection)[0]
            warn(f"Tried to execute an illegal action, executing {action} instead")

        if self.game_state.table_card == Constants.null_card_number:
            self.game_state.table_card = self.game_state.pop_card_of_agent(self.agent_selection, action)
            self.zero_out_reward()
        else:
            first_card = self.game_state.table_card
            second_card = self.game_state.pop_card_of_agent(self.agent_selection, action)

            self.game_state.add_seen_cards([first_card, second_card], [self.other_player(self.agent_selection), self.agent_selection])

            hand_points = get_cards_points([first_card, second_card])
            if is_first_player_win(first_card, second_card, get_seed(self.game_state.briscola_card)):
                winner = self.other_player(self.agent_selection)
            else:
                winner = self.agent_selection
                self.invert_player_turn()

            self.rewards[winner] = hand_points / Constants.total_points
            self.game_state.agent_points[winner] += hand_points

            self.game_state.table_card = Constants.null_card_number
            self.deal_cards(1, winner)

        self.next_turn()

        self._cumulative_rewards[self.agent_selection] = 0
        self._accumulate_rewards()

    def other_player(self, agent: str) -> str:
        return self.agents[0] if agent == self.agents[1] else self.agents[1]

    def next_turn(self):
        self.game_state.num_moves += 1
        self.game_state.current_agent = self.other_player(self.game_state.current_agent)

    def invert_player_turn(self):
        self.game_state.current_agent = self.other_player(self.game_state.current_agent)

    def render(self) -> str:
        return self.game_state.__repr__()

    def state(self) -> State:
        # Not used
        pass

    def deal_cards(self, n_cards: int, winner: Optional[str] = None) -> None:
        if winner is not None:
            agents = [winner, self.other_player(winner)]
        else:
            agents = self.agents

        for agent in agents:
            cards = self.game_state.extract_cards(n_cards)
            self.game_state.add_cards_to(agent, cards)

    def close(self):
        super().close()

    def is_over(self):
        return self.game_state.num_moves >= Constants.deck_cards

    def is_even(self):
        return self.game_state.agent_points[self.agents[0]] == self.game_state.agent_points[self.agents[1]]

    def game_winner(self) -> Optional[str]:
        if self.is_even():
            return None
        return sorted(self.game_state.agent_points.items(), key=lambda x: x[1], reverse=True)[0][0]

    def get_game_outcome(self, agent: str) -> float:
        if self.is_over():
            if self.is_even():
                return 0.5
            else:
                return 1. if self.game_winner() == agent else 0.
        return 0.

    def set_state(self, state: State):
        self.game_state = state

    def __repr__(self) -> str:
        return self.game_state.__repr__()
