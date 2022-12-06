import os
import sys
os.chdir(os.getcwd() + '/gym-platform')
sys.path.insert(0, os.getcwd())
from gym_platform.envs.platform_env import PlatformEnv, Constants
import numpy as np
from wrappers.wrappers import ScaledActionWrapper, ScaledStateWrapper, RewardWrapper, RllibWrapper


# Rllib requires environment to accept a dictionary env_config as input
def env_creator(env_config):
    """

    :param env_config: Dictionary as expected by Ray[rllib]
    :return: Wrapped environment with all necessary changes
    for our environment to work with Ray[rllib], state and action space
    """
    return ScaledActionWrapper(ScaledStateWrapper(RllibWrapper(PlatformEnv(), env_config)))


# ToDo: Use ray Tune to auto tune RL agents or Hyperopt/optuna to both tune and indicate parameter importance

def train(agent, training_itr=3000, validation_episodes=150, itr_per_validation=300, val=False, save_path=None):
    """

    :param agent: Compiled Agent to train
    :param training_itr: Number of training episodes
    :param validation_episodes: After every "iterations per validation", validate performance of the agent
    :param itr_per_validation: Number iterations to wait before validating agent's performance
    :param val: Whether or not to include validation in Training
    :return: Trained Agent nad Results of training
    :param save_path: Path to save the trained agent
    """
    results = None
    # Training Loop
    print('Training Starts:')
    for itr in range(training_itr):
        # collecting number of victories in each validation, this score should increase over the time
        vic_perc_training = []
        # Evaluate after every fixed iteration(validation)
        if val:
            if (itr + 1) % itr_per_validation == 0:
                # Currently cannot visualize in docker container
                if itr == 0:
                    print('This is how the agent behaves before training')
                curr_vic_perc = test_agent(total_episodes=validation_episodes, agent=agent, vis=False)
                vic_perc_training.append(curr_vic_perc)
                print(f'validation victory rate is {curr_vic_perc}%')
        print(f'Interation {itr} completed')
        results = agent.train()
    agent.save(save_path)
    return agent, results


# Testing
def test_agent(total_episodes, agent=None, vis=False):
    """
    :param vis: If visualizaing
    :param agent: RL Agent to test on current environment
    :param total_episodes: Total testing/validation episodes
    :return: Percentage of Episodes in which agent won
    """
    # create a test Environment
    test_env = env_creator({})
    collect_rewards = []
    for ep in range(total_episodes):
        sum_reward = 0
        state = test_env.reset()
        done = False
        while not done:
            action = agent.compute_action(state)
            state, reward, done, info = test_env.step(action)
            sum_reward += reward
            if vis:
                test_env.render()
        collect_rewards.append(sum_reward)
    collect_rewards = np.array(collect_rewards)
    return (collect_rewards[collect_rewards > 0.95].size * 100) / total_episodes
