# gym-snakegame
A [gymnasium](https://github.com/Farama-Foundation/Gymnasium)-based RL environment for learning the snake game.

![snakegame5x5](https://user-images.githubusercontent.com/54899900/225275703-646f8f42-62dd-4b0d-b527-aaf93ba6e1f7.gif)

You can make a game with a grid size larger than 5x5.

| Action Space | Observation Space |
|--------------|-------------------|
|  Discrete(4) | (`size`, `size`)  |

# Installation
```bash
git clone https://github.com/helpingstar/gym-snakegame.git
cd gym-snakegame
pip install -r requirements.txt
pip install -e .
```

## Colab

```bash
!git clone https://github.com/helpingstar/gym-snakegame.git
%cd gym-snakegame
!pip install -r requirements.txt
!pip install -e .
```

# Usage
```python
import gym_snakegame
import gymnasium as gym

env = gym.make('gym_snakegame/SnakeGame-v0', size=15, n_target=1, render_mode='rgb_array')
# env = gym.wrappers.RecordVideo(env, video_folder='./video_folder', episode_trigger=lambda x: x % 200 == 0)

observation, info = env.reset()
for i in range(100000):
    action = env.action_space.sample()
    obs, reward, terminated, _, info = env.step(action)
    if terminated:
        env.reset()
env.close()
```
## parameter
* `board_size` : The size of a square board. The board has the shape `(board_size, board_size)`.
* `n_target` : The number of targets placed on a board.

# Observation

Observation Space : `Box(0.0, board_size ** 2 + 1, (board_size, board_size), uint32)`

* `0` : empty
* `1 ~ board_size ** 2` : snake body
  * `1` : head
  * largest number : tail
* `board_size ** 2 + 1` : target


You can change the `dtype` or `shape` by using the wrapper below.

* [`gymnasium.experimental.wrappers.ReshapeObservationV0`](https://gymnasium.farama.org/api/experimental/wrappers/#gymnasium.experimental.wrappers.ReshapeObservationV0).
* [`gymnasium.experimental.wrappers.DtypeObservationV0`](https://gymnasium.farama.org/api/experimental/wrappers/#gymnasium.experimental.wrappers.DtypeObservationV0)

```python
import numpy as np
import gymnasium as gym
from gymnasium.experimental.wrappers import ReshapeObservationV0, DtypeObservationV0

import gym_snakegame

env = gym.make("gym_snakegame/SnakeGame-v0", board_size=5, n_target=1)
# Box(0, 26, (5, 5), uint32)
env = ReshapeObservationV0(env, (1, env.get_wrapper_attr('board_size'), env.get_wrapper_attr('board_size')))
# Box(0, 26, (1, 5, 5), uint32)
env = DtypeObservationV0(env, dtype=np.float32)
# Box(0.0, 26.0, (1, 5, 5), float32)
```


# Action

Action Space : `Discrete(4)`

* `0` : down
* `1` : right
* `2` : up
* `3` : left
