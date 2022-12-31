from gymnasium.envs.registration import register

register(
     id="gym_snakegame/SnakeGame-v0",
     entry_point="gym_snakegame.envs:SnakeGameEnv",
     max_episode_steps=300,
)
