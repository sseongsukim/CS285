from typing import Tuple
import numpy as np


class RandomAgent:
    def __init__(self, observation_shape: Tuple[int, ...], num_actions: int):
        super().__init__()
        self.num_actions = num_actions

    def get_action(self, *args, **kwargs):
        # TODO(student): Return a random action
        action = np.random.randint(low=0, high=self.num_actions)
        return action

    def update(self, *args, **kwargs):
        # Update is a no-op for the random agent
        return {}
