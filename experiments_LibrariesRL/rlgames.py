
'''
- El proceso de aprendizaje y la evaluacion son deterministas.
- La metrica a partir de los datos train asignada a cada politica: el cumulative reward del ultimo 
episodio completado en la secuencia continua de trajectorias (el episodio no tiene porque haberse completado 
en su totalidad por la politica a la que se le esta asignando la metrica).
- La politica output es la que menor valor de la metrica tiene entre las almacenadas con frecuencias constante
como checkpoint.

TODO: tengo que mirar a ver esta misma metrica como se interpreta cuando se ejecutan multiples entornos en paralelo.

TODO: el sats_window_size=1 de arriba puede que no sea siempre asi. Igual tiene que ver con:
self.games_num = self.minibatch_size // self.seq_length. Comprobarlo!!!
'''

from libraries.rlgames import Options
from libraries.commun import compress_decompress_list, training_stats
import pandas as pd
import numpy as np

# Ejecutando proceso
method='a2c_discrete'
env='PongNoFrameskip-v4'
seed=1
total_timesteps=128*16*10
experiment_name='execution1'
experiment_param= 'ppo_pong.yaml'
library_dir='/home/jesusangel/Dropbox/PhD/Mi trabajo/Codigo/OptimalResourceAllocation_RL/experiments_LibrariesRL/results/rlgames'
Options.learn_process(method,env,seed,total_timesteps,experiment_name,experiment_param,library_dir)
Options.learn_process(method,env,seed,200*1*10,'execution2',experiment_param,library_dir,
                      n_steps_per_env=200,n_workers=1,batch_size=50,save_best_after=1,save_frequency=1)
Options.learn_process(method,env,seed,200*1*10,'execution3',experiment_param,library_dir,
                      n_steps_per_env=200,n_workers=1,batch_size=50,save_best_after=1,save_frequency=1)

# Evaluando politicas
policy_id='nn/PongNoFrameskip_ray.pth'
Options.eval_policy(seed,5,experiment_name,experiment_param,library_dir,policy_id)
policy_id='process_info/policy9.pth'
Options.eval_policy(seed,5,experiment_name,experiment_param,library_dir,policy_id)
policy_id='process_info/policy7.pth'
Options.eval_policy(seed,5,experiment_name,experiment_param,library_dir,policy_id)

# Entendiendo output
df_traj=pd.read_csv('experiments_LibrariesRL/results/rlgames/execution3/process_info/df_traj.csv')
df_traj['traj_rewards']=[np.array(compress_decompress_list(i,compress=False)) for i in list(df_traj['traj_rewards'])]
df_traj['traj_ep_end']=[np.array(compress_decompress_list(i,compress=False)) for i in list(df_traj['traj_ep_end'])]


all_traj_rw=[]
all_traj_ep_end=[]
for k in range(10):
    all_traj_rw+=[j[0]for i in list(df_traj[df_traj['n_policy']==k+1]['traj_rewards'])[0] for j in i]
    all_traj_ep_end+=[i[0] for i in list(df_traj[df_traj['n_policy']==k+1]['traj_ep_end'])[0]]

print(training_stats(all_traj_rw,all_traj_ep_end,range(200,200*11,200),1))



        





