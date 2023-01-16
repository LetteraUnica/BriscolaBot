import sys
from time import sleep
from typing import Optional, Union

import pygame
from pygame.rect import Rect, RectType

from src.envs.two_player_briscola.BriscolaConstants import Constants
from src.envs.two_player_briscola.utils import get_seed, get_rank
from src.ui.Constants import UIConstants
from src.ui.controller.BriscolaController import BriscolaController


def get_card_path(card: Optional[int]) -> str:
    if card is None:
        return f"resources/briscola_cards/back.png"
    if card == Constants.null_card_number:
        return f"resources/briscola_cards/null_card.png"
    seed, rank = get_seed(card), get_rank(card)
    card_string = f"{rank + 1}_{UIConstants.seed_to_string[seed]}"
    return f"resources/briscola_cards/{card_string}.png"


def load_card_image(card: int) -> pygame.Surface:
    card_path = get_card_path(card)
    card_image = pygame.image.load(card_path)
    return card_image


def draw_card(screen: pygame.Surface, card: Optional[int], location: tuple[int, int]):
    card_image = load_card_image(card)
    card_image = pygame.transform.scale(card_image, UIConstants.card_size)
    screen.blit(card_image, location)


def draw_deck(screen: pygame.Surface):
    deck_image = pygame.image.load("resources/deck.png")
    deck_image = pygame.transform.scale(deck_image, UIConstants.card_size)
    location = (UIConstants.padding, (UIConstants.height - UIConstants.card_height) // 2)
    screen.blit(deck_image, location)


def draw_briscola_card(screen: pygame.Surface, briscola_card: int):
    location = (UIConstants.padding + UIConstants.card_width, (UIConstants.height - UIConstants.card_height) // 2)
    draw_card(screen, briscola_card, location)


def draw_table_cards(screen: pygame.Surface, cards: list[int]):
    for i, card in enumerate(cards):
        x = UIConstants.width - UIConstants.padding - UIConstants.card_width - i * (
                UIConstants.card_width + UIConstants.space_between_cards)
        y = (UIConstants.height - UIConstants.card_height) // 2
        draw_card(screen, card, (x, y))


def draw_human_hand(screen: pygame.Surface, human_cards: list[int]) -> list[Union[RectType, Rect]]:
    card_rects = []
    for i, card in enumerate(human_cards):
        x = UIConstants.width - UIConstants.padding - UIConstants.card_width - i * (
                UIConstants.card_width + UIConstants.space_between_cards)
        y = UIConstants.height - UIConstants.padding - UIConstants.card_height
        card_image = load_card_image(card)
        card_image = pygame.transform.scale(card_image, UIConstants.card_size)
        screen.blit(card_image, (x, y))
        card_rects.append(card_image.get_rect(topleft=(x, y)))

    return card_rects


def draw_ai_hand(screen: pygame.Surface, cards: list[int]):
    for i in range(len(cards)):
        x = UIConstants.width - UIConstants.padding - UIConstants.card_width - i * (
                UIConstants.card_width + UIConstants.space_between_cards)
        y = UIConstants.padding
        draw_card(screen, None, (x, y))


if __name__ == "__main__":
    controller = BriscolaController()
    pygame.init()
    screen = pygame.display.set_mode(UIConstants.size)
    pygame.display.set_caption("Briscola")

    card_rects = draw_human_hand(screen, controller.get_player_cards(UIConstants.human_player))
    second_played_card = None
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                sys.exit()

            mouse_pos = pygame.mouse.get_pos()
            for i, card_rect in enumerate(card_rects):
                if card_rect.collidepoint(mouse_pos) and event.type == pygame.MOUSEBUTTONDOWN:
                    card_clicked = controller.get_player_cards(UIConstants.human_player)[i]

                    controller.play_card(card_clicked)

        screen.fill(UIConstants.background_color)
        draw_briscola_card(screen, controller.get_briscola_card())
        draw_deck(screen)
        card_rects = draw_human_hand(screen, controller.get_player_cards(UIConstants.human_player))
        draw_ai_hand(screen, controller.get_player_cards(UIConstants.ai_player))
        draw_table_cards(screen, [Constants.null_card_number, controller.get_table_card()])

        sleep(0.03)
        pygame.display.flip()
