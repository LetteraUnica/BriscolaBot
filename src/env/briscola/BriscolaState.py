from dataclasses import dataclass
from random import sample, shuffle
from typing import List, Any, Dict

from src.env.briscola.BriscolaConstants import Constants
from src.env.briscola.Card import Card


def other_player(player: int) -> int:
    return (player + 1) % 2


def create_deck() -> List[Card]:
    deck = list(range(Constants.deck_cards))
    shuffle(deck)

    return [Card(card) for card in deck]


@dataclass
class BriscolaState:
    deck: List[Card]
    seen_cards: List[Card]
    player_cards: List[List[Card]]
    player_points: List[int]
    table_card: Card
    briscola_seed: int
    player_order: List[int]
    current_player: int

    def __init__(self):
        self.seen_cards = []
        self.reset()

    def reset(self):
        self.deck = create_deck()
        self.player_cards = [self.extract_cards(Constants.hand_cards), self.extract_cards(Constants.hand_cards)]
        self.player_points = [0, 0]
        self.player_order = sample(range(Constants.n_players), k=Constants.n_players)
        self.current_player = 0
        self.briscola_seed = self.get_briscola_card().get_seed()
        self.reset_table_card()

    def extract_cards(self, n_cards: int) -> List[Card]:
        cards_to_extract = min(n_cards, len(self.deck))
        extracted_cards = [self.deck.pop() for _ in range(cards_to_extract)]
        self.seen_cards.extend(extracted_cards)
        return extracted_cards

    def get_public_observation(self) -> Dict[str, Any]:
        return {"seen_cards": self.seen_cards.copy(),
                "table_card": self.get_table_card(),
                "briscola_card": self.get_briscola_card()}

    def get_player_observation(self, player_index: int) -> Dict[str, Any]:
        assert player_index in [0, 1]
        private_observation = self.get_public_observation()
        private_observation["hand_cards"] = self.player_cards[player_index].copy()
        return private_observation

    def get_current_player_observation(self) -> Dict[str, Any]:
        return self.get_player_observation(self.get_current_player())

    def get_current_player(self):
        return self.player_order[self.current_player]

    def set_table_card(self, card_played: Card):
        self.table_card = card_played

    def next_player(self):
        self.current_player = (self.current_player + 1) % 2

    def no_card_on_table(self) -> bool:
        return self.table_card.is_null()

    def pop_current_player_card(self, card_index: int) -> Card:
        return self.player_cards[self.get_current_player()].pop(card_index)

    def get_current_player_card(self, card_index: int) -> Card:
        return self.player_cards[self.get_current_player()][card_index]

    def deal_cards(self):
        for i in self.player_order:
            self.player_cards[i].extend(self.extract_cards(1))

    def reverse_player_order(self):
        self.player_order.reverse()

    def add_points(self, points: int, player_won: int):
        self.player_points[self.player_order[player_won]] += points

    def get_table_card(self) -> Card:
        return self.table_card

    def get_briscola_card(self) -> Card:
        if len(self.deck) > 0:
            return self.deck[-1]
        return Card(Constants.null_card_number)

    def get_briscola_seed(self) -> int:
        return self.briscola_seed

    def reset_table_card(self):
        self.table_card = Card(Constants.null_card_number)

    def get_first_player(self) -> int:
        return self.player_order[0]

    def no_card_in_player_hand(self) -> bool:
        return sum([len(cards) for cards in self.player_cards]) == 0
