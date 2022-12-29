from src.env.two_player_briscola.BriscolaConstants import Constants


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
