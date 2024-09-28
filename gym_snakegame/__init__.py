from gymnasium.envs.registration import register

__version__ = "0.0.1"

register(
     id="gym_snakegame/SnakeGame-v0",
     entry_point="gym_snakegame.envs:SnakeGameEnv",
)
