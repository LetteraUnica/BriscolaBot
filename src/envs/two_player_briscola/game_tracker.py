from typing import Optional, Any

import pandas as pd
from pandas import DataFrame
from torch import Tensor
import polars as pl

from src.envs.two_player_briscola.BriscolaConstants import Constants
from src.envs.two_player_briscola.TwoPlayerBriscola import TwoPlayerBriscola


class GameTracker(TwoPlayerBriscola):
    def __init__(self):
        super().__init__()
        self.things_to_track: dict[str, Any] = {}
        self.reset()

    def reset(self, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None) -> None:
        super().reset(seed, return_info, options)
        things_to_track = {
            "briscola_card": self.game_state.briscola_card,
        }
        self.things_to_track = things_to_track

    def step(self, action: int) -> None:
        self.track(action)
        super().step(action)

    def track(self, action: int):
        turn = self.game_state.num_moves
        for player, cards in self.game_state.hand_cards.items():
            for i, card in enumerate(cards):
                self.things_to_track[f"hand_card_{i}_{player}_{turn}"] = card
            for i in range(len(cards), Constants.hand_cards):
                self.things_to_track[f"hand_card_{i}_{player}_{turn}"] = Constants.null_card_number
        self.things_to_track[f"action_{turn}"] = action
        self.things_to_track[f"agent_to_play_{turn}"] = self.game_state.current_agent
        self.things_to_track[f"card_on_table_{turn}"] = self.game_state.table_card

    def to_df(self) -> DataFrame:
        return pl.DataFrame(self.things_to_track)
