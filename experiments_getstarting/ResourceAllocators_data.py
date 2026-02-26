'''
Este experimento esta ejecutado con una version anterior de our_library.py, por eso las .csv guardados no tienen la ultima 
columna comprimida ni en su nombre la coletilla "df_test_".
'''

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
from our_library import PPOLearner

list_seeds=list(range(1,41))

for seed in list_seeds:
    process=PPOLearner.start_learn_process(env=gym.make("InvertedPendulum-v4"),
                                    seed=seed,
                                    total_timesteps=200000,
                                    n_test_episodes=100, 
                                    path='experiments_getstarting/results/ResourceAllocators/InvertedPendulum/',
                                    csv_name='InvertedPendulum_seed'+str(seed)+'.csv')
    
for seed in list_seeds:
    process=PPOLearner.start_learn_process(env=gym.make("InvertedDoublePendulum-v4"),
                                    seed=seed,
                                    total_timesteps=500000,
                                    n_test_episodes=100, 
                                    path='experiments_getstarting/results/ResourceAllocators/InvertedDoublePendulum/',
                                    csv_name='InvertedDaublePendulum_seed'+str(seed)+'.csv')

    



