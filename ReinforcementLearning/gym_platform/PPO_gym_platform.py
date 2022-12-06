import argparse
from train_test import test_agent, train, env_creator
from configuration import config
import ray
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    n_itr_train = 700
    n_episodes_val = 50
    n_itr_val = 100
    n_episodes_test = 500
    assert isinstance(n_itr_train, int)
    assert isinstance(n_episodes_val, int)
    assert isinstance(n_itr_val, int)
    assert isinstance(n_episodes_test, int)
    checkpoint_path = "../PPO_gym_platform_checkpoint"
    env = "Platform-v0"

    # Register env with above name
    register_env(env, env_creator)

    # Initialize Ray
    ray.init()

    # Configure PPO agent with above config and environment
    trainer = PPOTrainer(env="Platform-v0", config=config)

    # Training the PPO agent
    trainer, _ = train(agent=trainer, training_itr=n_itr_train, validation_episodes=n_episodes_val,
                       itr_per_validation=n_itr_val, val=True, save_path=checkpoint_path)

    # Test Agent
    vic_perc_test = test_agent(total_episodes=n_episodes_test, agent=trainer)
    print(f'Agent won {vic_perc_test}% of it\'s matches in Testing ')

    # ToDo: Explainable AI using SHAP (https://github.com/slundberg/shap) or PCA.(currently visualizaed in Tensorboard)
    # Todo: A seperate training, testing, environment setup folder structure if using a complicated setup
    ray.shutdown()
