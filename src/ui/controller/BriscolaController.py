from src.envs.two_player_briscola.BriscolaConstants import Constants
from src.envs.two_player_briscola.TwoPlayerBriscola import TwoPlayerBriscola


class BriscolaController:
    def __init__(self):
        self.briscola_env = TwoPlayerBriscola()
        self.briscola_env.reset()

    def get_current_player(self) -> str:
        return self.briscola_env.agent_selection

    def get_player_cards(self, player: str) -> list[int]:
        return self.briscola_env.game_state.hand_cards[player]

    def get_table_card(self) -> int:
        return self.briscola_env.game_state.table_card

    def get_briscola_card(self) -> int:
        return self.briscola_env.game_state.briscola_card

    def play_card(self, card: int):
        self.briscola_env.step(card)

    def get_points_of_player(self, player: str) -> float:
        return self.briscola_env.game_state.agent_points[player]

    def is_over(self) -> bool:
        return self.briscola_env.is_over()
