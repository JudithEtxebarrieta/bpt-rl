from Main import *


data_path='experiments/results/data'
setups=[    # Classic Control
            'SB3_PPO_Pendulum',
            # Box2D
            'SB3_PPO_LunarLanderContinuous',
            'SB3_PPO_BipedalWalker',
            # MuJoCo
            'SB3_PPO_Swimmer',
            'SB3_PPO_HalfCheetah',
            'SB3_PPO_Hopper',
            'SB3_PPO_Ant',
            'SB3_PPO_Walker2d'            
                            ]

df = pd.DataFrame(columns=['setup', 'a_mean', 'a_std', 'b_mean', 'b_std'])

for setup in setups:

    df_regions=pd.read_csv(data_path+'/'+setup+'/learning_regions.csv')

    a_perc = (df_regions['a'] / df_regions['T'])*100
    b_perc = (df_regions['b'] / df_regions['T'])*100

    df.loc[len(df)] = [setup] + [round(i,2) for i in [np.mean(a_perc), np.std(a_perc), np.mean(b_perc),np.std(b_perc)]]

df.to_csv(data_path+'/paper/df_regions_perc_mean_std.csv')