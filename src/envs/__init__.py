from gymnasium.envs.registration import register

register(
    id="SinglePlayerBriscola-v0",
    entry_point="src.envs.single_player_briscola.SinglePlayerBriscola:SinglePlayerBriscola"
)
