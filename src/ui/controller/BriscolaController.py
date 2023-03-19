import onnxruntime as ort
import numpy as np
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

    def get_player_cards(self, player: str):
        player_cards = self.briscola_env.game_state.hand_cards[player].copy()
        try:
            player_cards.remove(self.played_card)
        except ValueError:
            pass
        return player_cards

    def get_table_cards(self):
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
            obs = self.briscola_env.observe(self.get_current_player())
            inputs = np.concatenate((obs["observation"], obs["action_mask"]), dtype=np.float32)
            actions = self.ai_policy.run(["action"], {"input": inputs.reshape(1, -1)})
            self.play_card(actions[0][0])

    def two_cards_on_table(self) -> bool:
        return Constants.null_card_number not in self.get_table_cards()

    def _load_ai_policy(self):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        session = ort.InferenceSession("src/ui/resources/agent.onnx", sess_options=opts)

        return session

    def get_winner(self) -> str:
        if self.is_over():
            return self.briscola_env.game_winner()

    def get_winner_points(self) -> float:
        if self.is_over():
            return self.get_points_of_player(self.get_winner())

    def get_deck_size(self) -> int:
        return len(self.briscola_env.game_state.deck)
