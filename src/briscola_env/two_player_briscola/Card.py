from dataclasses import dataclass

from src.env.two_player_briscola.BriscolaConstants import Constants
from src.env.two_player_briscola.utils import get_rank, get_seed


@dataclass
class Card:
    card_number: int

    def get_seed(self) -> int:
        return get_seed(self.card_number)

    def get_rank(self) -> int:
        return get_rank(self.card_number)

    def get_priority(self) -> int:
        return Constants.card_priorities[self.get_rank()]

    def get_points(self) -> int:
        return Constants.card_points[self.get_rank()]

    def is_null(self):
        return self.card_number == Constants.null_card_number

    def __repr__(self) -> str:
        if self.card_number == Constants.null_card_number:
            return f"NA"
        return f"{self.get_rank() + 1}{Constants.seed_representation[self.get_seed()]}"
