import numpy as np
import torch
from pettingzoo import AECEnv
from torch import Tensor

from src.agents import Agent
from src.envs.two_player_briscola.TwoPlayerBriscola import TwoPlayerBriscola

from scipy.special import logit

from src.vectorizers.VectorizedEnv import VectorizedEnv


def get_state_representation(envs: VectorizedEnv) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    obs = np.empty((len(envs),) + envs.single_observation_space()["observation"].shape, dtype=np.float32)
    action_masks = np.empty((len(envs), envs.single_action_space().n), dtype=np.int64)
    rewards = np.empty(len(envs), dtype=np.float32)
    dones = np.empty(len(envs), dtype=np.int8)
    for i, (observation, reward, termination, _, _) in enumerate(envs.last()):
        obs[i] = observation["observation"]
        action_masks[i] = observation["action_mask"]
        rewards[i] = reward
        dones[i] = termination

    return torch.tensor(obs), torch.tensor(action_masks), torch.tensor(rewards), torch.tensor(dones)


def play_all_moves_of_player(envs: list[AECEnv], policy: Agent, player: str, device: str = "cpu"):
    for _ in range(2):
        envs_to_play = [env for env in envs
                        if env.agent_selection == player
                        and not env.terminations[player]]
        if len(envs_to_play) == 0:
            break

        obs, action_mask = [], []
        for env in envs_to_play:
            observation = env.observe(env.agent_selection)
            obs.append(observation["observation"])
            action_mask.append(observation["action_mask"])

        obs, action_mask = np.array(obs, dtype=np.float32), np.array(action_mask, dtype=np.int64)
        actions = policy.get_actions(torch.tensor(obs).to(device), torch.tensor(action_mask).to(device))

        [env.step(action) for env, action in zip(envs_to_play, actions)]


def play_all_moves_of_players(vec_env: VectorizedEnv, policies: list[Agent], player: str, device="cpu"):
    n_envs, n_policies = len(vec_env), len(policies)
    for i, policy in enumerate(policies):
        start, end = (i * n_envs) // n_policies, ((i + 1) * n_envs) // n_policies
        play_all_moves_of_player(vec_env[start:end], policy, player, device)


def compute_rating(current_policy: Agent,
                   other_player_policy: Agent,
                   n_games: int = 512,
                   device: str = "cpu",
                   n_steps: int = 20,
                   env_fn=lambda: TwoPlayerBriscola(),
                   current_player: str = "player_0",
                   other_player: str = "player_1"):
    vec_env = VectorizedEnv(env_fn, n_games)
    vec_env.reset()
    other_player_policies = [other_player_policy]
    for _ in range(n_steps):
        with torch.no_grad():
            play_all_moves_of_players(vec_env, other_player_policies, other_player)
            next_obs, action_mask, _, _ = get_state_representation(vec_env)
            actions = current_policy.get_actions(next_obs.to(device), action_mask.to(device))

        vec_env.step(actions.cpu().numpy())
    play_all_moves_of_players(vec_env, other_player_policies, "player_1")  # Play the last move

    scores = np.array([env.get_game_outcome(current_player) for env in vec_env.get_envs()], dtype=np.float64)
    mean_score = np.mean(scores)
    return mean_score, logit(mean_score)
