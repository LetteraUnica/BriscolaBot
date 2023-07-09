import numpy as np
import torch

from src.agents.Agent import Agent
from src.agents.NNAgent import NNAgent
from src.envs.two_player_briscola.TwoPlayerBriscola import TwoPlayerBriscola
from src.envs.two_player_briscola.game_tracker import GameTracker
from src.utils.training_utils import play_all_moves_of_players
from src.vectorizers.VectorizedEnv import VectorizedEnv


def track_games(current_policy: Agent,
                other_player_policy: Agent,
                n_games: int = 512,
                device: str = "cpu",
                env_fn=lambda: TwoPlayerBriscola(),
                current_player: str = "player_0",
                other_player: str = "player_1",
                n_hands: int = 20):
    vec_env = VectorizedEnv(env_fn, n_games)
    vec_env.reset()
    for _ in range(n_hands + 1):
        with torch.no_grad():
            play_all_moves_of_players(vec_env, [other_player_policy], other_player, device=device)
            play_all_moves_of_players(vec_env, [current_policy], current_player, device=device)

    scores = np.array([env.get_game_outcome(current_player) for env in vec_env.get_envs()], dtype=np.float64)
    return vec_env, (np.mean(scores)), np.std(scores) / np.sqrt(scores.shape[0])


if __name__ == "__main__":
    device = "cpu"
    observation_shape = (162,)
    action_size = 40

    trained_previous = NNAgent(observation_shape, action_size, hidden_size=256).to(device)
    trained_previous.load_state_dict(torch.load("pretrained_models/briscola-bot-v3.pt"))

    envs, score, scorestd = track_games(trained_previous,
                              trained_previous,
                              n_games=2048,
                              env_fn=lambda: GameTracker())

    print(score, scorestd)
    print([len(env.actions) for env in envs.get_envs()])
