# Tuned Hyperparameters
config = {'gamma': 0.9,  # working gamma=0.94
          'lr': 1e-4,  # 'lr': 1e-3,
          'num_workers': 0,
          "rollout_fragment_length": 64,  # 64 #both works
          'train_batch_size': 256,  # 256 #both works
          'sgd_minibatch_size': 32,  # 64 #both works
          "normalize_actions": False,
          'batch_mode': 'truncate_episodes',
          "vf_clip_param": 10,
          "entropy_coeff": 0.1,
          "entropy_coeff_schedule": [
              [0, 0.1],
              [150000, 0.001],
          ],
          'model': {
              'fcnet_hiddens': [64, 64, 32],  # 128,64,64,32
              'fcnet_activation': 'relu'
          }
          }