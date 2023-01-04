from src.envs.two_player_briscola.BriscolaConstants import Constants


def get_seed(card: int) -> int:
    return card // Constants.cards_per_seed


def get_rank(card: int) -> int:
    return card % Constants.cards_per_seed


def get_points(card: int) -> float:
    return Constants.card_points[get_rank(card)]


def get_priority(card: int, hand_seed: int, briscola_seed: int) -> int:
    seed, rank = get_seed(card), get_rank(card)

    if seed == briscola_seed:
        return Constants.cards_per_seed * Constants.card_priorities[rank] + 1

    if seed == hand_seed:
        return Constants.card_priorities[rank]

    return 0


def is_first_player_win(first_card: int, second_card: int, briscola_seed: int) -> bool:
    hand_seed = get_seed(first_card)
    return get_priority(first_card, hand_seed, briscola_seed) > get_priority(second_card, hand_seed, briscola_seed)


def get_cards_points(cards: list[int]) -> int:
    return sum([get_points(card) for card in cards])
