class UIConstants:
    brighten: int = 64
    text_color: tuple[int, int, int] = (255, 255, 255)
    background_color: tuple[int, int, int] = (34, 139, 34)  # Forest green

    ui_scale: float = 0.8

    small_font_size: int = int(30 * ui_scale)
    big_font_size: int = int(60 * ui_scale)

    width: int = int(800 * ui_scale)
    height: int = int(800 * ui_scale)
    padding: int = int(20 * ui_scale)
    size: tuple[int, int] = (width, height)

    card_width: int = int(150 * ui_scale)
    card_height: int = int(225 * ui_scale)
    space_between_cards: int = int(30 * ui_scale)
    card_size: tuple[int, int] = (card_width, card_height)

    seed_to_string: dict[int, str] = {0: "bastoni",
                                      1: "coppe",
                                      2: "denari",
                                      3: "spade"}
    players: tuple[str, str] = ("player_0", "player_1")
    human_player: str = players[0]
    ai_player: str = players[1]
