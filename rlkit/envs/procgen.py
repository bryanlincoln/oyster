import gym
from gym import spaces
import numpy as np
import torch
from . import register_env


@register_env("procgen")
class ProcgenEnv:
    def __init__(
        self,
        n_tasks=60,
        n_train_tasks=50,
        n_test_tasks=10,
        seed=0,
        distribution_mode="easy",
        env_names="procgen:procgen-coinrun-v0",
        **kwargs
    ):
        self._seed = seed
        self.n_tasks = n_tasks
        self.n_train_tasks = n_train_tasks
        self.n_test_tasks = n_test_tasks
        self.tasks = []
        self._goal = 0
        self.distribution_mode = distribution_mode
        self.env_names = env_names

        self._sample_tasks()  # fill up self.tasks
        self.env = self.tasks[0]  # this env will change depending on the idx

        self.observation_space = self.env.observation_space

        # action space is Discrete(15)
        # we need to transform it to continuous
        # so we use an one-hot approach
        self.action_space = spaces.Box(low=0, high=1, shape=(15,))

    def step(self, action):
        return self.env.step(action.argmax())

    def reset(self):
        # apparently, this env doesn't implement a proper reset method
        # so we have to send a -1 as action
        # see: https://github.com/openai/procgen/issues/40#issuecomment-633720234
        obs, _, _, _ = self.env.step(-1)
        return obs

    def seed(self, _seed):
        self._seed = _seed
        self._sample_tasks()

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self.env = self.tasks[idx]
        self._goal = idx
        self.reset()

    def render(self):
        self.env.render()

    def _sample_tasks(self):
        self.tasks = [
            gym.make(self.env_names, start_level=idx, num_levels=1, distribution_mode=self.distribution_mode)
            for idx in range(self._seed, self._seed + self.n_tasks)
        ]
