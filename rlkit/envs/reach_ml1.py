import numpy as np

from metaworld.benchmarks import ML1
from . import register_env


@register_env('reach-ml1')
class ReachML1Env():

    def __init__(self, task={}, n_train_tasks=50, n_test_tasks=10, randomize_tasks=True, out_of_distribution=False, **kwargs):
        self.train_env = ML1.get_train_tasks('reach-v1', out_of_distribution=out_of_distribution)
        self.test_env = ML1.get_test_tasks('reach-v1', out_of_distribution=out_of_distribution)
        self.train_tasks = self.train_env.sample_tasks(n_train_tasks)
        self.test_tasks = self.test_env.sample_tasks(n_test_tasks)
        self.tasks = self.train_tasks + self.test_tasks
        self.env = self.train_env #this env will change depending on the idx
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.reset_task(0)

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def seed(self, seed):
        self.train_env.seed(seed)
        self.test_env.seed(seed)

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        if idx < len(self.train_tasks):
            self.env = self.train_env
            self.env.set_task(self.train_tasks[idx])
            self._goal = self.train_tasks[idx]['goal']
        else:
            self.env = self.test_env
            idx = idx - len(self.train_tasks)
            self.env.set_task(self.test_tasks[idx])
            self._goal = self.test_tasks[idx]['goal']

    def render(self):
        self.env.render()

