import sys
from functools import lru_cache
from time import sleep
from typing import Optional, Union, List, Tuple

import pygame
from pygame.rect import Rect, RectType

from src.envs.two_player_briscola.BriscolaConstants import Constants
from src.envs.two_player_briscola.utils import get_seed, get_rank
from src.ui.UIConstants import UIConstants
from src.ui.controller.BriscolaController import BriscolaController


@lru_cache(maxsize=None)
def get_card_path(card: Optional[int]) -> str:
    if card is None:
        return "src/ui/resources/briscola_cards/back.png"
    if card == Constants.null_card_number:
        return "src/ui/resources/briscola_cards/empty.png"
    seed, rank = get_seed(card), get_rank(card)
    card_string = f"{rank + 1}_{UIConstants.seed_to_string[seed]}"
    return f"src/ui/resources/briscola_cards/{card_string}.png"


@lru_cache(maxsize=None)
def load_card_image(card: int) -> pygame.Surface:
    card_path = get_card_path(card)
    card_image = pygame.image.load(card_path)
    return rescale(card_image)


@lru_cache(maxsize=None)
def rescale(image: pygame.Surface) -> pygame.Surface:
    return pygame.transform.smoothscale(image, (UIConstants.card_width, UIConstants.card_height))


def draw_card(screen: pygame.Surface, card: Optional[int], location: Tuple[int, int]):
    card_image = load_card_image(card)
    screen.blit(card_image, location)


@lru_cache(maxsize=None)
def load_deck_image() -> pygame.Surface:
    deck_image = pygame.image.load("src/ui/resources/deck.png")
    return rescale(deck_image)


def draw_deck(screen: pygame.Surface, deck_cards: int):
    deck_image = load_deck_image()
    location = (UIConstants.padding, (UIConstants.height - UIConstants.card_height) // 2)
    screen.blit(deck_image, location)

    font = pygame.font.Font(None, UIConstants.small_font_size)
    screen.blit(
        font.render(f'Remaining cards: {deck_cards}', True, UIConstants.text_color, UIConstants.background_color),
        (UIConstants.padding,
         (UIConstants.height - UIConstants.card_height) // 2 + UIConstants.card_height + UIConstants.padding)
    )


def draw_briscola_card(screen: pygame.Surface, briscola_card: int):
    location = (UIConstants.padding + UIConstants.card_width, (UIConstants.height - UIConstants.card_height) // 2)
    draw_card(screen, briscola_card, location)


def draw_table_cards(screen: pygame.Surface, cards: List[int]):
    for i, card in enumerate(cards):
        x = UIConstants.width - UIConstants.padding - UIConstants.card_width - i * (
                UIConstants.card_width + UIConstants.space_between_cards)
        y = (UIConstants.height - UIConstants.card_height) // 2
        draw_card(screen, card, (x, y))


def draw_human_hand(screen: pygame.Surface, human_cards: List[int]) -> List[Union[RectType, Rect]]:
    card_rects = []
    is_over_card = 0
    for i, card in enumerate(human_cards):
        x = UIConstants.width - UIConstants.padding - UIConstants.card_width - i * (
                UIConstants.card_width + UIConstants.space_between_cards)
        y = UIConstants.height - UIConstants.padding - UIConstants.card_height
        card_image = load_card_image(card).copy()
        card_rect = card_image.get_rect(topleft=(x, y))
        if card_rect.collidepoint(pygame.mouse.get_pos()):
            card_image.fill((255, 255, 128, 192), special_flags=pygame.BLEND_RGBA_MULT)
            is_over_card += 1

        screen.blit(card_image, (x, y))
        card_rects.append(card_rect)

    if is_over_card > 0:
        pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
    else:
        pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)

    return card_rects


def draw_ai_hand(screen: pygame.Surface, cards: List[int]):
    for i in range(len(cards)):
        x = UIConstants.width - UIConstants.padding - UIConstants.card_width - i * (
                UIConstants.card_width + UIConstants.space_between_cards)
        y = UIConstants.padding
        draw_card(screen, None, (x, y))


def print_points(screen: pygame.Surface, player_points: float, ai_points: float):
    font = pygame.font.Font(None, UIConstants.small_font_size)
    screen.blit(
        font.render(f'AI points: {ai_points}', True, UIConstants.text_color, UIConstants.background_color),
        (2 * UIConstants.padding, 2 * UIConstants.padding)
    )

    screen.blit(
        font.render(f'Player points: {player_points}', True, UIConstants.text_color, UIConstants.background_color),
        (2 * UIConstants.padding, UIConstants.height - 3 * UIConstants.padding)
    )


def print_win_screen(screen: pygame.Surface, player_won: Optional[str], points: float):
    pygame.draw.rect(screen, UIConstants.background_color, (0, 0, screen.get_width(), screen.get_height()))

    font = pygame.font.Font(None, UIConstants.big_font_size)
    if player_won is None:
        win_text = "It's a draw!"
    else:
        player_won = "You" if player_won == UIConstants.human_player else "AI"
        win_text = f'{player_won} won with {points} points!'
     
    text = font.render(win_text,
                       True,
                       UIConstants.text_color)
    text_rect = text.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2))
    screen.blit(text, text_rect)


if __name__ == "__main__":
    controller = BriscolaController()
    pygame.init()
    screen = pygame.display.set_mode(UIConstants.size)
    pygame.display.set_caption("Briscola")

    card_rects = draw_human_hand(screen, controller.get_player_cards(UIConstants.human_player))
    while True:
        delay = 0.
        is_over_card = 0
        card_clicked = None
        for event in pygame.event.get():
            if (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE) or event.type == pygame.QUIT:
                sys.exit()

            for i, card_rect in enumerate(card_rects):
                if card_rect.collidepoint(
                        pygame.mouse.get_pos()) and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    card_clicked = controller.get_player_cards(UIConstants.human_player)[i]

        if card_clicked is not None:
            controller.play_card(card_clicked)
        controller.play_ai_card()

        if controller.two_cards_on_table():
            delay += 2.

        screen.fill(UIConstants.background_color)
        card_rects = draw_human_hand(screen, controller.get_player_cards(UIConstants.human_player))
        draw_ai_hand(screen, controller.get_player_cards(UIConstants.ai_player))
        draw_table_cards(screen, controller.get_table_cards())
        draw_briscola_card(screen, controller.get_briscola_card())
        if controller.get_deck_size() > 0:
            draw_deck(screen, controller.get_deck_size())
        print_points(screen,
                     controller.get_points_of_player(UIConstants.human_player),
                     controller.get_points_of_player(UIConstants.ai_player))

        if controller.is_over():
            print_win_screen(screen, controller.get_winner(), controller.get_winner_points())
            controller.reset()
            delay += 2.

        pygame.display.flip()
        sleep(delay + 0.03)
        controller.next_tick()
