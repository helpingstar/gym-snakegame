import numpy as np
from collections import deque
import pygame

import gymnasium as gym
from gymnasium import spaces


# TODO : literal observation to CONST VARIABLE

SNAKE_BODY = 0
BLANK = 1
SNAKE_HEAD = 3
TARGET = 5

"""
[x, y]
    x : row
    y : column

snake
    [index]
    head : -1
    tail : 0
    append : append
    pop : popleft

[value] : board
    0 : snake
    1 : blank
    3 : head
    5 : target
"""
class SnakeGameEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "ansi", "None"], "render_fps": 10}

    def __init__(self, render_mode=None, size=15, n_target=1):

        assert size >= 5
        assert n_target > 0

        self.size = size  # The size of the square grid
        self.window_width = 600  # The size of the PyGame window
        self.window_height = 700
        self.window_diff = self.window_height - self.window_width
        self.n_target = n_target
        self._n_iteration = 0
        # space
        self.observation_space = spaces.Box(
            low=0, high=5, shape=(size, size), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        self._snake_checker = np.ones((size, size), dtype=np.uint8)

        assert render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # initialize board
        self.board = np.ones((self.size, self.size), dtype=np.uint8)

        # initialize snake
        self.snake = deque()
        for i in range(3):
            self.snake.appendleft(np.array([self.size//2, self.size//2-i]))

        for x, y in self.snake:
            self.board[x, y] = SNAKE_BODY
        self.board[self.snake[-1][0], self.snake[-1][1]] = SNAKE_HEAD

        self._place_target(initial=True)

        # update iteration
        self._n_iteration += 1
        self._n_step = 0

        self._score = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _place_target(self, initial=False):
        target_candidate = np.argwhere(self.board == 1)
        if initial:
            target_list = target_candidate[self.np_random.choice(len(target_candidate), self.n_target)]
            for x, y in target_list:
                self.board[x, y] = TARGET
        else:
            if target_candidate.size == 0:
                return
            else:
                new_target = target_candidate[self.np_random.choice(len(target_candidate))]
                self.board[new_target[0], new_target[1]] = TARGET


    def _get_obs(self):
        return self.board.astype(np.float32)

    def _get_info(self):
        return {"snake_length": len(self.snake)}

    def step(self, action):

        direction = self._action_to_direction[action]

        # update iteration
        self._n_step += 1

        current_head = self.snake[-1]
        current_tail = self.snake[0]
        next_head = current_head + direction

        if np.array_equal(next_head, self.snake[-2]):
            next_head = current_head - direction

        # get out the board
        if not (0 <= next_head[0] < self.size and 0 <= next_head[1] < self.size):
            reward = -1
            terminated = True

        # hit the snake
        elif self.board[next_head[0], next_head[1]] == 0:
            reward = -1
            terminated = True
        else:

            # blank
            if self.board[next_head[0], next_head[1]] == 1:
                self.board[current_tail[0], current_tail[1]] = 1
                self.snake.popleft()
                reward = 0
                terminated = False
            # target
            # self.board[next_head[0], next_head[1]] == 5
            else:
                self._score += 1
                reward = 1
                self._place_target()
                terminated = False
            self.snake.append(next_head)
            self.board[current_head[0], current_head[1]] = SNAKE_BODY
            self.board[next_head[0], next_head[1]] = SNAKE_HEAD


        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):

        if self.render_mode == "None":
            return

        if self.render_mode == "rgb_array":
            return self._render_frame()

        if self.render_mode == "ansi":
            return self._render_ansi()

    def _render_ansi(self):
        for r in range(self.size):
            for c in range(self.size):
                # snake
                if self.board[r, c] == SNAKE_BODY:
                    print("□", end="")
                # blank
                elif self.board[r, c] == BLANK:
                    print("0", end="")
                # head
                elif self.board[r, c] == SNAKE_HEAD:
                    print("■", end="")
                # target
                elif self.board[r, c] == TARGET:
                    print("★", end="")
            print()
        print('-'*self.size)


    def _render_frame(self):
        pygame.font.init()
        if self.window is None:
            pygame.init()
            self.square_size = self.window_width // self.size
            self.font_size = self.window_diff // 3
            if self.render_mode == "human":
                pygame.display.init()
                # (width, height)
                self.window = pygame.display.set_mode(
                    (self.window_width, self.window_height)
                )
            else:
                self.window = pygame.Surface((self.window_width, self.window_height))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((0, 0, 0))
        myFont = pygame.font.SysFont('consolas', self.font_size, bold=True)
        score_render_text = myFont.render(f'score: {self._score}', True, (255, 255, 255))
        n_iter_render_text = myFont.render(f'iter: {self._n_iteration}', True, (255, 255, 255))
        n_step_render_text = myFont.render(f'step: {self._n_step}', True, (255, 255, 255))

        canvas.blit(score_render_text, (self.window_width // 30 * 1, self.window_diff // 2 - self.font_size // 2))
        canvas.blit(n_iter_render_text, (self.window_width // 30 * 10, self.window_diff // 2 - self.font_size // 2))
        canvas.blit(n_step_render_text, (self.window_width // 30 * 21, self.window_diff // 2 - self.font_size // 2))


        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] == SNAKE_BODY:
                    pygame.draw.rect(
                        canvas,
                        (255, 255, 255),
                        pygame.Rect(
                            self.square_size * c, self.window_diff + self.square_size * r, self.square_size, self.square_size
                        ),
                    )
                # blank
                elif self.board[r, c] == BLANK:
                    pygame.draw.rect(
                        canvas,
                        (200, 200, 200),
                        pygame.Rect(
                            self.square_size * c, self.window_diff + self.square_size * r, self.square_size, self.square_size
                        ),
                        1
                    )
                # head
                elif self.board[r, c] == SNAKE_HEAD:
                    pygame.draw.rect(
                        canvas,
                        (255, 0, 0),
                        pygame.Rect(
                            self.square_size * c, self.window_diff + self.square_size * r, self.square_size, self.square_size
                        ),
                    )
                # self.board[r, c] == 5:
                else:
                    pygame.draw.rect(
                        canvas,
                        (0, 255, 0),
                        pygame.Rect(
                            self.square_size * c, self.window_diff + self.square_size * r, self.square_size, self.square_size
                        ),
                    )

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