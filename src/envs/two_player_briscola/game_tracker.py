from typing import Optional, Any

from src.envs.two_player_briscola.TwoPlayerBriscola import TwoPlayerBriscola, State


class GameTracker(TwoPlayerBriscola):
    def __init__(self):
        super().__init__()
        self.things_to_track: dict[str, Any] = {}
        self.actions: list[int] = []
        self.states: list[State] = []
        self.reset()

    def step(self, action: int) -> None:
        self.states.append(self.get_game_state())
        self.actions.append(action)
        super().step(action)

    def reset(self, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None) -> None:
        super().reset(seed, return_info, options)
        things_to_track = {
            "briscola_card": self.game_state.briscola_card,
            "p1_hand_cards": [],
            "p2_hand_cards": [],
            "actions": [],
            "agent_to_play": [],
        }
        self.things_to_track = things_to_track
