from Main import *
import gymnasium as gym
import pandas as pd


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

data_regions=False
data_envs=True


if data_regions:
    df = pd.DataFrame(columns=['setup', 'a_mean', 'a_std', 'b_mean', 'b_std'])

    for setup in setups:

        df_regions=pd.read_csv(data_path+'/'+setup+'/learning_regions.csv')

        a_perc = (df_regions['a'] / df_regions['T'])*100
        b_perc = (df_regions['b'] / df_regions['T'])*100

        df.loc[len(df)] = [setup] + [round(i,2) for i in [np.mean(a_perc), np.std(a_perc), np.mean(b_perc),np.std(b_perc)]]

    df.to_csv(data_path+'/paper/df_regions_perc_mean_std.csv')

if data_envs:
    env_names = [
        "LunarLanderContinuous-v3",
        "BipedalWalker-v3",
        "Swimmer-v4",
        "Hopper-v4",
        "HalfCheetah-v4",       
        "Walker2d-v4",
        "Ant-v4"

    ]

    def data_env_complexity(env_name):

        def detect(env):
            u = env.unwrapped
            if hasattr(u, "world"):
                return "Box2D"
            elif hasattr(u, "model") and hasattr(u, "data"):
                return "MuJoCo"
            else:
                return "Unknown"
            
        data = []

        for env_name in env_names:

            env = gym.make(env_name)

            engine=detect(env)

            if engine=='Box2D':

                _, _ = env.reset()
                unwrapped = env.unwrapped
                world = unwrapped.world

                data.append({
                    "env": env_name,
                    'engine': engine,
                    "dim_s": env.observation_space.shape[0],
                    "dim_a": env.action_space.shape[0],
                    "bodies": len(list(world.bodies)),
                    "joints": len(list(world.joints)),
                })


            if engine=='MuJoCo':

                env.reset()

                model = env.unwrapped.model

                data.append({
                    "env": env_name,
                    'engine': engine,
                    "dim_s": env.observation_space.shape[0],
                    "dim_a": env.action_space.shape[0],
                    "bodies": model.nbody,
                    "joints": model.njnt,
                })

                env.close()


        df = pd.DataFrame(data)
        df.to_csv(data_path+'/paper/df_env_complexity.csv')

    data_env_complexity(env_names)