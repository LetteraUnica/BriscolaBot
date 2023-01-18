class UIConstants:
    brighten = 64
    text_color: tuple[int, int, int] = (255, 255, 255)
    background_color: tuple[int, int, int] = (34, 139, 34)  # Forest green

    width: int = 800
    height: int = 800
    padding: int = 20
    size: tuple[int, int] = (width, height)

    card_width: int = 150
    card_height: int = 225
    space_between_cards: int = 30
    card_size: tuple[int, int] = (card_width, card_height)

    seed_to_string: dict[int, str] = {0: "bastoni",
                                      1: "coppe",
                                      2: "denari",
                                      3: "spade"}

    players: tuple[str, str] = ("player_0", "player_1")
    human_player: str = players[0]
    ai_player: str = players[1]
