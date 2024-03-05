from typing import Any, SupportsFloat
import gymnasium as gym
from gym_snakegame.envs.snake_game import SnakeGameEnv
import numpy as np


class SnakeActionMask(gym.Wrapper):
    def __init__(self, env: SnakeGameEnv, mask_wall: bool = True, mask_snake: bool = True):
        super().__init__(env)
        self.action_to_direction = env.get_wrapper_attr("_action_to_direction")
        self.ITEM = env.get_wrapper_attr("ITEM")
        self.board_size = env.get_wrapper_attr("board_size")
        self.snake = env.get_wrapper_attr("snake")
        self.board = env.get_wrapper_attr("board")
        self.valid_pos = (0, self.ITEM)
        self.mask_wall = mask_wall
        self.mask_snake = mask_snake

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        info["action_mask"] = self.get_action_mask()
        return obs, info

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        info["action_mask"] = self.get_action_mask()
        return obs, reward, terminated, truncated, info

    def get_action_mask(self):
        r, c = self.snake[-1]
        action_mask = np.zeros(4, dtype=np.int8)
        for i, (dr, dc) in enumerate(self.action_to_direction):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                if self.board[nr][nc] in self.valid_pos:
                    action_mask[i] = 1
                else:
                    if not self.mask_snake:
                        action_mask[i] = 1
            else:
                if not self.mask_wall:
                    action_mask[i] = 1
        return action_mask
