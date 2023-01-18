import torch

from src.agents.NNAgent import NNAgent
from src.envs.two_player_briscola.BriscolaConstants import Constants
from src.envs.two_player_briscola.TwoPlayerBriscola import TwoPlayerBriscola
from src.ui.UIConstants import UIConstants


class BriscolaController:
    def __init__(self):
        self.played_card = None
        self.briscola_env = TwoPlayerBriscola()
        self.ai_player = UIConstants.ai_player
        self.ai_policy = self._load_ai_policy()

        self.reset()

    def reset(self):
        self.played_card = Constants.null_card_number
        self.briscola_env.reset()

    def get_current_player(self) -> str:
        return self.briscola_env.agent_selection

    def get_player_cards(self, player: str) -> list[int]:
        player_cards = self.briscola_env.game_state.hand_cards[player].copy()
        try:
            player_cards.remove(self.played_card)
        except ValueError:
            pass
        return player_cards

    def get_table_cards(self) -> list[int]:
        return [self.played_card, self.briscola_env.game_state.table_card]

    def get_briscola_card(self) -> int:
        return self.briscola_env.game_state.briscola_card

    def play_card(self, card: int):
        if not self.is_over():
            if self.briscola_env.game_state.table_card == Constants.null_card_number:
                self.briscola_env.step(card)
            else:
                self.played_card = card

    def get_points_of_player(self, player: str) -> float:
        return self.briscola_env.game_state.agent_points[player]

    def is_over(self) -> bool:
        return self.briscola_env.is_over()

    def next_tick(self):
        if self.played_card != Constants.null_card_number:
            self.briscola_env.step(self.played_card)
        self.played_card = Constants.null_card_number

    def play_ai_card(self):
        if self.get_current_player() == UIConstants.ai_player:
            with torch.no_grad():
                obs = self.briscola_env.observe(self.get_current_player())
                observation, action_mask = torch.tensor(obs["observation"]), torch.tensor(obs["action_mask"])
                actions = self.ai_policy.get_actions(observation.view(1, -1), action_mask.view(1, -1))
            self.play_card(actions[0].cpu().item())

    def two_cards_on_table(self) -> bool:
        return Constants.null_card_number not in self.get_table_cards()

    def _load_ai_policy(self):
        player_policy = NNAgent(self.briscola_env.observation_space(self.ai_player)["observation"].shape,
                                self.briscola_env.action_space(self.ai_player).n)

        player_policy.load_state_dict(torch.load("src/ui/resources/agent.pt"))
        player_policy.eval()

        return player_policy

    def get_winner(self) -> str:
        if self.is_over():
            return self.briscola_env.game_winner()

    def get_winner_points(self) -> float:
        if self.is_over():
            return self.get_points_of_player(self.get_winner())

    def get_deck_size(self) -> int:
        return len(self.briscola_env.game_state.deck)
