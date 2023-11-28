import numpy as np
from collections import deque
import pygame

import gymnasium as gym
from gymnasium import spaces


class SnakeGameEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array", "ansi"],
        "render_fps": 20,
    }

    def __init__(self, render_mode=None, n_channel=1, board_size=15, n_target=1):
        assert board_size >= 5
        assert n_target > 0
        assert n_channel in (1, 2, 4)

        self.BLANK = 0
        self.ITEM = board_size**2 + 1
        self.HEAD = 1
        self.n_channel = n_channel

        self.color_gradient = (255 - 100) / (board_size**2)

        self.board_size = board_size  # The size of the square grid
        self.window_width = 600  # The size of the PyGame window
        self.window_height = 700
        self.window_diff = self.window_height - self.window_width
        self.n_target = n_target
        # space
        self.observation_space = spaces.Box(
            low=0,
            high=self.ITEM,
            shape=(self.n_channel, board_size, board_size),
            dtype=np.uint32,
        )
        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([1, 0]),  # down
            1: np.array([0, 1]),  # right
            2: np.array([-1, 0]),  # up
            3: np.array([0, -1]),  # left
        }

        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # initialize board
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.uint32)

        # initialize snake
        self.snake = deque()
        for i in range(3):
            self.snake.appendleft(
                np.array([self.board_size // 2, self.board_size // 2 - i])
            )
        for i, (x, y) in enumerate(self.snake):
            self.board[x, y] = len(self.snake) - i

        self._place_target(initial=True)

        # update iteration
        self._n_step = 0
        self._score = 0
        self.prev_action = 1

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, info

    def _place_target(self, initial: bool = False) -> None:
        target_candidate = np.argwhere(self.board == self.BLANK)
        if initial:
            target_list = target_candidate[
                self.np_random.choice(len(target_candidate), self.n_target)
            ]
            for x, y in target_list:
                self.board[x, y] = self.ITEM
        else:
            if target_candidate.size == 0:
                return
            else:
                new_target = target_candidate[
                    self.np_random.choice(len(target_candidate))
                ]
                self.board[new_target[0], new_target[1]] = self.ITEM

    def _get_obs(self):
        if self.n_channel == 1:
            return self.board
        else:
            return self._split_channel(self.n_channel)

    def _split_channel(self, n_channel):
        if n_channel == 2:
            mask = self.board == self.ITEM
            snake_obs = np.where(mask, 0, self.board)
            target_obs = np.where(mask, self.board, 0)
            return np.array([snake_obs, target_obs])
        # n_channel == 4
        else:
            channels = []
            # body
            mask = (1 < self.board) & (self.board < len(self.snake))
            channel = np.where(mask, self.board, 0)
            channels.append(channel)

            # head, tail, target
            without_body = (1, len(self.snake), self.ITEM)
            for element in without_body:
                mask = self.board == element
                channel = np.where(mask, self.board, 0)
                channels.append(channel)

            return np.array(channels)

    def _get_info(self):
        return {"snake_length": len(self.snake), "prev_action": self.prev_action}

    def step(self, action: int):
        direction = self._action_to_direction[action]

        # update iteration
        self._n_step += 1

        current_head = self.snake[-1]
        current_tail = self.snake[0]
        next_head = current_head + direction

        if np.array_equal(next_head, self.snake[-2]):
            next_head = current_head - direction

        # get out the board
        if not (
            0 <= next_head[0] < self.board_size and 0 <= next_head[1] < self.board_size
        ):
            reward = -1
            terminated = True
        # hit the snake
        elif 0 < self.board[next_head[0], next_head[1]] < self.ITEM:
            reward = -1
            terminated = True
        else:
            # blank
            if self.board[next_head[0], next_head[1]] == self.BLANK:
                self.board[current_tail[0], current_tail[1]] = self.BLANK
                self.snake.popleft()
                reward = 0
                terminated = False
            # target
            # self.board[next_head[0], next_head[1]] == self.ITEM
            else:
                self._score += 1
                reward = 1
                self._place_target()
                self.board[next_head[0], next_head[1]] = 0
                if len(self.snake) == self.board_size**2:
                    terminated = True
                else:
                    terminated = False
            self.snake.append(next_head)
            for x, y in self.snake:
                self.board[x][y] += 1

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        self.prev_action = action

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        if self.render_mode in {"rgb_array", "human"}:
            return self._render_frame()

    def get_body_color(self, r: int, c: int):
        color = 255 - self.color_gradient * self.board[r][c]
        return (color, color, color)

    def _render_frame(self):
        pygame.font.init()
        if self.window is None:
            pygame.init()
            self.square_size = self.window_width // self.board_size
            self.font_size = self.window_diff // 3
            if self.render_mode == "human":
                pygame.display.init()
                self.window = pygame.display.set_mode(
                    (self.window_width, self.window_height)
                )
            else:
                self.window = pygame.Surface((self.window_width, self.window_height))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((0, 0, 0))
        myFont = pygame.font.SysFont("consolas", self.font_size, bold=True)
        score_render_text = myFont.render(
            f"score: {self._score}", True, (255, 255, 255)
        )
        n_step_render_text = myFont.render(
            f"step: {self._n_step}", True, (255, 255, 255)
        )

        canvas.blit(
            score_render_text,
            (self.window_width // 30 * 1, self.window_diff // 2 - self.font_size // 2),
        )
        canvas.blit(
            n_step_render_text,
            (self.window_width // 30 * 15, self.window_diff // 2 - self.font_size // 2),
        )

        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[r, c] == self.BLANK:
                    pygame.draw.rect(
                        canvas,
                        (200, 200, 200),
                        pygame.Rect(
                            self.square_size * c,
                            self.window_diff + self.square_size * r,
                            self.square_size,
                            self.square_size,
                        ),
                        1,
                    )
                # head
                elif self.board[r, c] == self.HEAD:
                    pygame.draw.rect(
                        canvas,
                        (255, 0, 0),
                        pygame.Rect(
                            self.square_size * c,
                            self.window_diff + self.square_size * r,
                            self.square_size,
                            self.square_size,
                        ),
                    )
                elif self.board[r, c] == self.ITEM:
                    pygame.draw.rect(
                        canvas,
                        (0, 255, 0),
                        pygame.Rect(
                            self.square_size * c,
                            self.window_diff + self.square_size * r,
                            self.square_size,
                            self.square_size,
                        ),
                    )
                # body
                else:
                    pygame.draw.rect(
                        canvas,
                        self.get_body_color(r, c),
                        pygame.Rect(
                            self.square_size * c,
                            self.window_diff + self.square_size * r,
                            self.square_size,
                            self.square_size,
                        ),
                    )
                # blank

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
