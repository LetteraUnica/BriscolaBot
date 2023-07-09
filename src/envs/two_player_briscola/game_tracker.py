from copy import deepcopy
from typing import Optional, Any

import polars as pl

from src.envs.two_player_briscola.BriscolaConstants import Constants
from src.envs.two_player_briscola.TwoPlayerBriscola import TwoPlayerBriscola


class GameTracker(TwoPlayerBriscola):
    def __init__(self):
        super().__init__()
        self.tracked_metrics: dict[str, Any] = {}
        self.reset()

    def reset(self, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None) -> None:
        super().reset(seed, return_info, options)
        self.tracked_metrics = {
            "briscola_card": self.game_state.briscola_card,
        }

    def step(self, action: int) -> None:
        self.track(action)
        super().step(action)

    def track(self, action: int):
        turn = self.game_state.num_moves
        for player, cards in self.game_state.hand_cards.items():
            for i, card in enumerate(cards):
                self.tracked_metrics[f"hand_card_{i}_{player}_{turn}"] = card
            for i in range(len(cards), Constants.hand_cards):
                self.tracked_metrics[f"hand_card_{i}_{player}_{turn}"] = Constants.null_card_number
        self.tracked_metrics[f"action_{turn}"] = action
        self.tracked_metrics[f"agent_to_play_{turn}"] = self.game_state.current_agent
        self.tracked_metrics[f"card_on_table_{turn}"] = self.game_state.table_card

    def get_game_history(self) -> dict:
        return deepcopy(self.tracked_metrics)
