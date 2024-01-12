# gym-snakegame
A [gymnasium](https://github.com/Farama-Foundation/Gymnasium)-based RL environment for learning the snake game.



https://github.com/helpingstar/gym-snakegame/assets/54899900/4eca0135-ec4b-4a07-9815-51c1d3df1823



You can make a game with a grid size larger than 5x5.

| Action Space | Observation Space |
|--------------|-------------------|
|  Discrete(4) | (`board_size`, `board_size`)  |

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

env = gym.make(
    "gym_snakegame/SnakeGame-v0", board_size=10, n_channel=1, n_target=1, render_mode='human'
)

obs, info = env.reset()
for i in range(10000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```
## parameter
* `board_size` : The size of a square board. The board has the shape `(board_size, board_size)`.
* `n_target` : The number of targets placed on a board.
* `n_channel` : The number of channels of the observation. The observation has the shape `(n_channel, board_size, board_size)`.

# Observation

Observation Space : `Box(0.0, board_size ** 2 + 1, (n_channel, board_size, board_size), uint32)`

## Channel
* 1 : Like the rendering format, all information is expressed in one channel.
* 2 : The snake and item channels are divided.
  * 0 : snake
  * 1 : item
* 4 : Channels are divided in the following order.
  * 0 : snake's head
  * 1 : snake's body
  * 2 : snake's tail
  * 3 : item

## How to represent each element.
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
from gymnasium.experimental.wrappers import DtypeObservationV0

import gym_snakegame

env = gym.make("gym_snakegame/SnakeGame-v0", board_size=5, n_target=1)
# Box(0, 26, (1, 5, 5), uint32)
env = DtypeObservationV0(env, dtype=np.float32)
# Box(0.0, 26.0, (1, 5, 5), float32)
print(env.observation_space)
```


# Action

Action Space : `Discrete(4)`

* `0` : down
* `1` : right
* `2` : up
* `3` : left


## Invalid Action

When the Agent takes an Invalid Action, the Agent continues in the direction it was previously going.
