from copy import deepcopy
from typing import List

import torch
from torch import tensor, nn
from torch.distributions import Categorical

from src.agents.Agent import Agent
from src.envs.two_player_briscola.BriscolaConstants import Constants
from src.envs.two_player_briscola.TwoPlayerBriscola import TwoPlayerBriscola, State


def get_current_player_cards(briscola_env: TwoPlayerBriscola) -> List[int]:
    return briscola_env.game_state.hand_cards[briscola_env.agent_selection]


def find_best_move(briscola_env: TwoPlayerBriscola):
    player = briscola_env.agent_selection
    if briscola_env.is_over():
        return briscola_env.game_state.agent_points[player]

    best_move = None
    best_score = -1
    for card in get_current_player_cards(briscola_env):
        next_env = deepcopy(briscola_env)
        next_env.step(card)
        _, score = find_best_move(next_env)
        if score > best_score:
            best_score = score
            best_move = card
    return best_move, best_score


def get_env(observation):
    def card_indexes(cards: tensor) -> list[int]:
        return (cards > 0.1).nonzero().squeeze().tolist()

    thrown_cards = observation[:Constants.deck_cards]
    briscola_card = observation[Constants.deck_cards:Constants.deck_cards * 2]
    table_card = observation[Constants.deck_cards * 2:Constants.deck_cards * 3]
    player_cards = observation[Constants.deck_cards * 3:Constants.deck_cards * 4]
    player_points = observation[-2] * Constants.total_points
    opponent_points = observation[-1] * Constants.total_points

    opponent_cards = 1 - thrown_cards - player_cards
    state = State(deck=[],
                  thrown_cards_player=[],
                  thrown_cards=card_indexes(thrown_cards),
                  hand_cards={"player_0": card_indexes(player_cards), "player_1": card_indexes(opponent_cards)},
                  table_card=card_indexes(table_card)[0],
                  briscola_card=card_indexes(briscola_card)[0],
                  current_agent="player_0",
                  agent_points={"player_0": player_points, "player_1": opponent_points},
                  num_moves=thrown_cards.sum()
                  )

    env = TwoPlayerBriscola()
    env.set_state(state)
    return env


class SearchingAgent(nn.Module, Agent):
    def __init__(self, policy: nn.Module, action_size: int, name: str = "Searching-Agent"):
        super().__init__()
        self.actor = policy
        self.action_size = action_size
        self.name = name

    def get_name(self) -> str:
        return self.name

    def get_probs(self, observations, action_masks):
        observations = self.obs_transform(observations)
        logits = self.actor(observations)
        if action_masks is not None:
            logits[~action_masks.bool()] = -1e8
        probs = Categorical(logits=logits)
        return probs

    def get_actions(self, observations: tensor, action_masks: tensor = None):
        actions = torch.empty(observations.shape[0], dtype=torch.int64)
        for i, observation in enumerate(observations):
            if observation[:Constants.deck_cards].sum() == 0:
                actions[i] = find_best_move(get_env(observation))
            else:
                actions[i] = self.get_probs(observations[i].view(1, -1), action_masks[i].view(1, -1)).sample()
        return actions

    def forward(self, inputs: tensor):
        observation, action_mask = inputs[:, :-self.action_size], inputs[:, -self.action_size:]
        return self.get_actions(observation, action_mask)
