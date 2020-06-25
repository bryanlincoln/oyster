import gym
from gym import spaces
import numpy as np
import torch
from rlkit.torch.networks import CNN
from . import register_env


@register_env("coinrun")
class CoinRunEnv:
    def __init__(self, n_tasks=60, n_train_tasks=50, n_test_tasks=10, seed=0, use_obs_encoder=True, **kwargs):
        self._seed = seed
        self.n_tasks = n_tasks
        self.n_train_tasks = n_train_tasks
        self.n_test_tasks = n_test_tasks
        self.tasks = []
        self._sample_tasks()  # fill up self.tasks
        self.env = self.tasks[0]  # this env will change depending on the idx

        if use_obs_encoder:
            self.obs_encoder = CNN(channels=self.env.observation_space.shape[-1])
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.obs_encoder.output_size,)
            )
        else:
            self.obs_encoder = lambda x: x
            self.observation_space = self.env.observation_space

        # coinrun action space is Discrete(15)
        # we need to transform it to continuous
        # so we use an one-hot approach
        self.action_space = spaces.Box(low=0, high=1, shape=(15,))

    def step(self, action):
        obs, reward, done, info = self.env.step(action.argmax())
        return self._preprocess(obs), reward, done, info

    def reset(self):
        # apparently, this env doesn't implement a proper reset method
        # so we have to send a -1 as action
        # see: https://github.com/openai/procgen/issues/40#issuecomment-633720234
        obs, _, _, _ = self.env.step(-1)
        return self._preprocess(obs)

    def seed(self, _seed):
        self._seed = _seed
        self._sample_tasks()

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self.env = self.tasks[idx]
        self.reset()

    def render(self):
        self.env.render()

    def _sample_tasks(self):
        self.tasks = [
            gym.make("procgen:procgen-coinrun-v0", start_level=idx, num_levels=1)
            for idx in range(self._seed, self._seed + self.n_tasks)
        ]

    def _preprocess(self, obs):
        # add batch dimension -> instantiate tensor -> process -> remove batch dimension
        return self.obs_encoder(torch.tensor(obs[None], dtype=torch.float32))[0]
