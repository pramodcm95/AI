import os
import sys
import gym
from gym import spaces
import copy
from gym.spaces import Box, Tuple
import numpy as np


from gym_platform.envs.platform_env import Constants


# Wrappepr to change dtype of continuous states and action to float64 and add env_config to constructor for Rllib
class RllibWrapper(gym.Wrapper):
    def __init__(self, env, env_config):
        """

        :param env: Gym Platform environment the agent has to solve
        :param env_config -> dict: Requirement of ray[rllib], generally useful in Automatic Hyperparmeter tune
        """
        super().__init__(env)
        self.env = env
        self.env_config = env_config


# To scale or modify reward if necessary(not necessary here)
class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, rew):
        # modify rew
        return rew


class ScaledStateWrapper(gym.ObservationWrapper):
    """
    Scales the observation space to interval [-1,1]
    # Inspired or Adapted from "https://github.com/cycraig/MP-DQN/blob/master/common/wrappers.py"
    """

    def __init__(self, env):
        super(ScaledStateWrapper, self).__init__(env)
        obs = env.observation_space
        self.low = obs.spaces[0].low
        self.high = obs.spaces[0].high
        self.observation_space = Tuple(
            (gym.spaces.Box(low=-np.ones(self.low.shape), high=np.ones(self.high.shape),
                            dtype=np.float64),
             obs.spaces[1]))

    def observation(self, obs):
        state, steps = obs
        state = 2. * (state - self.low) / (self.high - self.low) - 1.
        ret = (state, steps)
        return ret


class ScaledActionWrapper(gym.ActionWrapper):
    """
    Changes the scale of the continuous action parameters to [-1,1].
    Parameter space not flattened in this case
    # Inspired or Adapted from "https://github.com/cycraig/MP-DQN/blob/master/common/wrappers.py"
    """

    def __init__(self, env):
        super(ScaledActionWrapper, self).__init__(env)
        self.actions_space_old = env.action_space
        self.num_actions = self.actions_space_old.spaces[0].n
        self.high = [self.actions_space_old.spaces[1][i].high for i in range(self.num_actions)]
        self.low = [self.actions_space_old.spaces[1][i].low for i in range(self.num_actions)]
        self.range = [self.actions_space_old.spaces[1][i].high - self.actions_space_old.spaces[1][i].low for i in
                      range(self.num_actions)]
        new_params = [  # parameters
            gym.spaces.Box(-np.zeros(self.actions_space_old.spaces[1][i].low.shape),
                           np.ones(self.actions_space_old.spaces[1][i].high.shape),
                           dtype=np.float64)
            for i in range(self.num_actions)
        ]
        self.action_space = gym.spaces.Tuple((
            self.actions_space_old.spaces[0],  # actions
            gym.spaces.Tuple(tuple(new_params)),
        ))

    def action(self, action):
        """
        Rescale to True Actions
        """
        action = copy.deepcopy(action)
        # A tuple cannot be assigned with a value
        action = list(action)
        action[1] = list(action[1])
        action[1][action[0]] = self.range[action[0]] * (action[1][action[0]] + 1) / 2. + self.low[action[0]]
        return action
