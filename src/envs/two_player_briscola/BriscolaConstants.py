from typing import Dict


class Constants:
    seed_representation: Dict[int, str] = {0: "C",
                                           1: "B",
                                           2: "S",
                                           3: "D"}
    card_priorities: Dict[int, int] = {0: 10,
                                       1: 1,
                                       2: 9,
                                       3: 2,
                                       4: 3,
                                       5: 4,
                                       6: 5,
                                       7: 6,
                                       8: 7,
                                       9: 8}
    card_points: Dict[int, int] = {0: 11,
                                   1: 0,
                                   2: 10,
                                   3: 0,
                                   4: 0,
                                   5: 0,
                                   6: 0,
                                   7: 2,
                                   8: 3,
                                   9: 4}
    total_points: int = 120
    cards_per_seed: int = 10
    n_agents: int = 2
    hand_cards: int = 3
    deck_cards: int = 40
    null_card_number: int = 40
