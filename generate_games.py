import re
import subprocess
from typing import Any

import numpy as np
import pytest
import torch
import polars as pl

import click
from numpy import ndarray

from src.agents.Agent import Agent
from src.agents.NNAgent import NNAgent
from src.envs.two_player_briscola.BriscolaConstants import Constants
from src.envs.two_player_briscola.TwoPlayerBriscola import TwoPlayerBriscola
from src.envs.two_player_briscola.game_tracker import GameTracker
from src.utils.training_utils import play_all_moves_of_players, concat_dicts
from src.vectorizers.VectorizedEnv import VectorizedEnv
from tqdm.auto import tqdm

HIDDEN_SIZE = 256
device = "cpu"
OBS_SHAPE = (162,)
ACTION_SIZE = 40


def track_games(current_policy: Agent,
                other_player_policy: Agent,
                n_games: int = 512,
                device: str = "cpu",
                env_fn=lambda: TwoPlayerBriscola(),
                current_player: str = "player_0",
                other_player: str = "player_1",
                n_hands: int = 20) -> tuple[VectorizedEnv, ndarray, Any]:
    vec_env = VectorizedEnv(env_fn, n_games)
    vec_env.reset()
    for _ in tqdm(range(n_hands + 1), "Playing Briscola tricks"):
        with torch.no_grad():
            play_all_moves_of_players(vec_env, [other_player_policy], other_player, device=device)
            play_all_moves_of_players(vec_env, [current_policy], current_player, device=device)

    scores = np.array([env.get_game_outcome(current_player) for env in vec_env.get_envs()], dtype=np.float64)
    return vec_env, (np.mean(scores)), np.std(scores) / np.sqrt(scores.shape[0])


@click.command()
@click.option("--games", default=2048, help="Number of games to play and track.")
@click.option("--fname", default="games.parquet", help="Output filename")
def generate_games(**kwargs) -> VectorizedEnv:
    print("Initializing environments")
    policy = NNAgent(OBS_SHAPE, ACTION_SIZE, hidden_size=HIDDEN_SIZE).to(device)
    policy.load_state_dict(torch.load("pretrained_models/briscola-bot-v3.pt"))

    tracked_envs, score, scorestd = track_games(policy,
                                                policy,
                                                n_games=kwargs["games"],
                                                env_fn=lambda: GameTracker())
    print(f"Played {kwargs['games']} games, score: {score} +- {scorestd}, sigma: {abs(score - 0.5) / scorestd}."
          f" The score should be close to 0.5")

    print(f"Saving games on {kwargs['fname']}")
    games_played = concat_dicts([tracked_env.get_game_history() for tracked_env in tracked_envs.get_envs()])
    pl.DataFrame(games_played).write_parquet(kwargs["fname"])


if __name__ == "__main__":
    generate_games()


@pytest.mark.parametrize("kwargs", [{"games": 32, "fname": "test_games.parquet"},
                                    {"games": 64, "fname": "test/test_games.parquet"}])
def test_generate_games(kwargs, capfd):
    process = subprocess.Popen(f"python generate_games.py --games={kwargs['games']} --fname={kwargs['fname']}",
                               shell=True)
    process.wait()

    # Capture printed output
    captured = capfd.readouterr()
    printed_output = captured.out
    # Verify the expected output is present
    sigma_match = re.search(r"sigma: ([+-]?([0-9]*[.])?[0-9]+)", printed_output)
    sigma_value = float(sigma_match.group(1))
    assert sigma_value < 5

    # check dataframe shape
    games = pl.read_parquet(kwargs["fname"])
    assert len(games.columns) == 9 * Constants.deck_cards + 1
    assert len([col for col in games.columns if "hand_card" in col]) == 6 * Constants.deck_cards
    assert len([col for col in games.columns if "action" in col]) == Constants.deck_cards
    assert len([col for col in games.columns if "agent_to_play" in col]) == Constants.deck_cards
    assert len([col for col in games.columns if "card_on_table" in col]) == Constants.deck_cards
    assert "briscola_card" in games.columns
    assert games.shape[0] == kwargs["games"]

    # Clean saved files
    process = subprocess.Popen(f"rm -rf {kwargs['fname']}", shell=True)
    process.wait()
