import gymnasium as gym
import numpy as np


class RewardConverter(gym.RewardWrapper):
    def __init__(self, env, moving_reward = 0):
        super().__init__(env)
        self.moving_reward = moving_reward

    def reward(self, reward):
        if reward == 0:
            return self.moving_reward
        else:
            return reward