"""Minimal sanity PPO training for the pure-RL baseline.

Single environment, small CNN over the perception grid, ~50k steps. The goal
is only to confirm the learning loop turns and reward trends up — not a final
run. Vectorized envs, eval callbacks, checkpoints and configs come later
"""

import os

import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.rl.env import SaREnv


class GridCNN(BaseFeaturesExtractor):
    """Small CNN ovre the (channels, H, W) perception grid."""

    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim) # Let the parent class know (SB3 needs this)
        n_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # stride halves the grid
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, *observation_space.shape)).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flat, features_dim), nn.ReLU()) # from 6400 feature vector to the desired output size

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations)) # the forward pass obs (B, C, H, W) -> features (B, features_dim) -> linear
                                                  # -> (B, features_dim) output for the policy/value net


# TODO: add eval callback, checkpoints, config, maybe vectorized envs if not too much hassle.
def main(total_timesteps: int = 50_000):
    env = Monitor(SaREnv())

    model = PPO(
        "MlpPolicy", 
        env,
        policy_kwargs=dict(
            features_extractor_class=GridCNN,
            features_extractor_kwargs=dict(features_dim=256),
            normalize_images=False,  # obs already in [0, 1]
        ),
        n_steps=1024,
        batch_size=64,
        ent_coef=0.01,
        verbose=1,
        seed=0,
    )

    model.learn(total_timesteps=total_timesteps)

    os.makedirs("runs", exist_ok=True)
    model.save("runs/ppo_sanity")
    print("saved model to runs/ppo_sanity.zip")


if __name__ == "__main__":
    main()
