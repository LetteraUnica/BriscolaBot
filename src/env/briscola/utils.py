from src.env.briscola.BriscolaConstants import Constants


def get_seed(card: int) -> int:
    return card // Constants.cards_per_seed


def get_rank(card: int) -> int:
    return card % Constants.cards_per_seed
