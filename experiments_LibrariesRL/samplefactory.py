'''
Entendiendo el output de PPO en sample-factory.

Se hacen ciertas comprobaciones, y se observa que:

1) Sobre funciones implementadas: 
- Sequential learn_process with samplefactory is deterministic: True
- Parallel learn_process with samplefactory is deterministic: False
- Default evaluation with samplefactory is deterministic: False
- My implementation eval_policy is deterministic: True

NOTE: el motivo de la estocasticidad en paralelo creo que es por el 'reset' en el environment. A pesar de fijar la semilla
para el environment, cuando n_workers>1, en diferentes ejecuciones la simulacion no tarda lo mismo, y el environment ejecutado
cada workers puede terminar el episodio antes o despues. Por eso, el tiempo en que se resetea el environment no tiene porque 
coincidir en dos ejecuciones diferentes, y los estados iniciales varian dependiendo de la ejecucion. En realidad la secuencia 
de estados iniciales en orden de reset repartida entre n_workers deberia de ser la misma, pero el i-esimo reset no siempre le
correspondera al mismo worker. Por eso, la secuencia de trayectorias y politicas cambia, aunque la forma de actualizar del 
learner siempre sea la misma porque en numpy y torch se fijan las semillas aleatorias.

2) Sobre politica output:
Parametros:
- save_every_sec
- save_best_every_sec
- save_best_metric
- stats_avg
- keep_checkpoints

La politica output es la ultima o mejor (especificado en save_best_metric) entre las keep_checkpoints almacenadas
como checkpoint con frecuencia periodica save_every_sec. Si se escoge la opcion save_best_metric='best', se monitoriza
entre las checkpoint la mejor cada save_best_every_sec, con respecto a la metrica save_best_metric. La unica implementada
es el 'reward', que es el reward promedio de los ultimos stats_avg episodios completos almacenados en train (entre la union
de las trayectorias durante el aprendizaje).

Cuando num_workers>1, como en cada worker se hace una interaccion independiente de la politica de longitud rollout, 
por interaccion se almacenan tantas trayectorias como workers. Si 
x= (batch_size*num_batches_per_epoch)//(rollout*num_workers*num_envs_per_worker)
el numero de interacciones que hay que repetir para tener suficientes datos para actualizar la politica es >1, las trayectorias
que se almacenan en process_info/df_traj.csv por iteracion son una matriz de tamaño (x*num_workers,rollout). He observado que
si num_workers=2, las filas impares de esa matriz son las trayectorias del worker 1 y las pares del worker 2. Supongo que para mas
workers las filas de la matriz seguiran el patron: fila1-worker1, fila2-worker2, fila3-worker3, fila4-worker1, fila5-worker2, fila6-worker6,...

'''
import numpy as np
import pandas as pd
from libraries.samplefactory import Options
from libraries.commun import compress_decompress_list, external_run, training_stats_single_worker, training_stats_multiple_workers

# Definir parametros
method='APPO'
env='mujoco_ant'
seed=1
total_timesteps=64*2*5 # batch_size*num_batches_per_epoch*(numero de iteraciones). rollout*num_workers*num_envs_per_worker 
                       # tiene que ser un multiplo de batch_size*num_batches_per_epoch, i.e., los datos generados durante la
                       # interaccion con el entorno deben ser exactamente los necesarios para definir un epoch en la actualizacion de la politica.
library_dir='experiments_LibrariesRL/results/samplefactory'

#--------------------------------------------------------------------------------------------------
# Proceso determinista-> SOLO EN EJECUCION SECUENCIAL
#--------------------------------------------------------------------------------------------------
# Secuencial
Options.learn_process(method,env,seed,total_timesteps,'execution11',library_dir,
                            n_steps_per_env=64,n_workers=1,n_envs_per_worker=1,
                            batch_size=64*2,n_batches_per_epoch=1,n_epoch=1)
Options.learn_process(method,env,seed,total_timesteps,'execution2',library_dir,
                            n_steps_per_env=64,n_workers=1,n_envs_per_worker=1,
                            batch_size=64*2,n_batches_per_epoch=1,n_epoch=1)
# Paralelo
Options.learn_process(method,env,seed,total_timesteps,'execution3',library_dir,
                            n_steps_per_env=64,n_workers=2,n_envs_per_worker=1,
                            batch_size=64*2,n_batches_per_epoch=1,n_epoch=1)
Options.learn_process(method,env,seed,total_timesteps,'execution4',library_dir,
                            n_steps_per_env=64,n_workers=2,n_envs_per_worker=1,
                            batch_size=64*2,n_batches_per_epoch=1,n_epoch=1)
external_run('experiments_LibrariesRL/samplefactory.py',range(48,74))

df_traj1=pd.read_csv('experiments_LibrariesRL/results/samplefactory/execution1/process_info/df_traj.csv')
df_traj1['traj_rewards']=[np.array(compress_decompress_list(i,compress=False)) for i in list(df_traj1['traj_rewards'])]
df_traj1['traj_ep_end']=[np.array(compress_decompress_list(i,compress=False)) for i in list(df_traj1['traj_ep_end'])]
df_traj2=pd.read_csv('experiments_LibrariesRL/results/samplefactory/execution2/process_info/df_traj.csv')
df_traj2['traj_rewards']=[np.array(compress_decompress_list(i,compress=False)) for i in list(df_traj2['traj_rewards'])]
df_traj2['traj_ep_end']=[np.array(compress_decompress_list(i,compress=False)) for i in list(df_traj2['traj_ep_end'])]
print('Sequential learn_process with samplefactory is deterministic: '+str(df_traj1.equals(df_traj2)))

df_traj1=pd.read_csv('experiments_LibrariesRL/results/samplefactory/execution2/process_info/df_traj.csv')
df_traj1['traj_rewards']=[np.array(compress_decompress_list(i,compress=False)) for i in list(df_traj1['traj_rewards'])]
df_traj1['traj_ep_end']=[np.array(compress_decompress_list(i,compress=False)) for i in list(df_traj1['traj_ep_end'])]
df_traj2=pd.read_csv('experiments_LibrariesRL/results/samplefactory/execution4/process_info/df_traj.csv')
df_traj2['traj_rewards']=[np.array(compress_decompress_list(i,compress=False)) for i in list(df_traj2['traj_rewards'])]
df_traj2['traj_ep_end']=[np.array(compress_decompress_list(i,compress=False)) for i in list(df_traj2['traj_ep_end'])]
print('Parallel learn_process with samplefactory is deterministic: '+str(df_traj1.equals(df_traj2)))

#--------------------------------------------------------------------------------------------------
# Evaluacion determinista-> SI
#--------------------------------------------------------------------------------------------------
# Por defecto
eval1=Options.eval_policy(env,seed,3,'execution1',library_dir,policy_id=0)
eval2=Options.eval_policy(env,seed,3,'execution1',library_dir,policy_id=0)
# Con mi modificacion
eval3=Options.eval_policy(env,seed,3,'execution1',library_dir,policy_id=0,deterministic_eval=True)
eval4=Options.eval_policy(env,seed,3,'execution1',library_dir,policy_id=0,deterministic_eval=True)

try:
    print('Default evaluation with samplefactory is deterministic: '+str(all(eval1[0]==eval2[0])))
    print('My implementation eval_policy is deterministic: '+str(all(eval3[0]==eval4[0])))
except:
    pass

external_run('experiments_LibrariesRL/samplefactory.py',list(range(48,56))+list(range(92,107)))

#--------------------------------------------------------------------------------------------------
# Entender cual es la politica output
#--------------------------------------------------------------------------------------------------
# Definir parametros para obligar a guardar en checkpointing todas las politicas (
# save_every_sec y save_best_every_sec lo mas pequeño posible, batch_size*n_batches_per_epoch lo suficientemente
# grande como para que una iteracion consuma mas de save_every_sec)
total_timesteps=64*10*5
Options.learn_process(method,env,seed,total_timesteps,'execution5',library_dir,
                            n_steps_per_env=64,n_workers=1,n_envs_per_worker=1,
                            batch_size=64*10,n_batches_per_epoch=1,n_epoch=1,
                            save_every_sec=1, keep_checkpoints=8,save_best_every_sec=1,save_best_after=total_timesteps)
external_run('experiments_LibrariesRL/samplefactory.py',list(range(48,56))+list(range(110,121)))

# Comprobar que las politicas son las que he guardado y cual es la que se selecciona como mejor
eval1=Options.eval_policy(env,seed,3,'execution5',library_dir,policy_id=0,deterministic_eval=True)
eval2=Options.eval_policy(env,seed,3,'execution5',library_dir,policy_id=1,deterministic_eval=True)
eval3=Options.eval_policy(env,seed,3,'execution5',library_dir,policy_id=2,deterministic_eval=True)
eval4=Options.eval_policy(env,seed,3,'execution5',library_dir,policy_id=3,deterministic_eval=True)
eval5=Options.eval_policy(env,seed,3,'execution5',library_dir,policy_id=4,deterministic_eval=True)
eval6=Options.eval_policy(env,seed,3,'execution5',library_dir,policy_id=5,deterministic_eval=True)
eval7=Options.eval_policy(env,seed,3,'execution5',library_dir,policy_id=6,deterministic_eval=True)
eval8=Options.eval_policy(env,seed,3,'execution5',library_dir,policy_id=7,deterministic_eval=True)

eval9=Options.eval_policy(env,seed,3,'execution5',library_dir,checkpoint_id='000000000_0',deterministic_eval=True)
eval10=Options.eval_policy(env,seed,3,'execution5',library_dir,checkpoint_id='000000002_1280',deterministic_eval=True)
eval12=Options.eval_policy(env,seed,3,'execution5',library_dir,checkpoint_id='000000003_1920',deterministic_eval=True)
eval13=Options.eval_policy(env,seed,3,'execution5',library_dir,checkpoint_id='000000004_2560',deterministic_eval=True)
eval14=Options.eval_policy(env,seed,3,'execution5',library_dir,checkpoint_id='000000006_3840',deterministic_eval=True)
eval15=Options.eval_policy(env,seed,3,'execution5',library_dir,checkpoint_id='000000007_4480',deterministic_eval=True)
eval16=Options.eval_policy(env,seed,3,'execution5',library_dir,deterministic_eval=True)
eval17=Options.eval_policy(env,seed,3,'execution5',library_dir,deterministic_eval=True,load_checkpoint_kind='best')

try:
    print('Validacion de politicas en secuencia')
    print(eval1)
    print(eval2)
    print(eval3)
    print(eval4)
    print(eval5)
    print(eval6)
    print(eval7)
    print(eval8)
    print('Validacion de politicas checkpointing')
    print(eval9)
    print(eval10)
    print(eval12)
    print(eval13)
    print(eval14)
    print(eval15)
    print(eval16)
    print(eval17)
except:
    pass

external_run('experiments_LibrariesRL/samplefactory.py',list(range(48,56))+list(range(123,163)))

# Como se calcula la metrica 'reward' a partil de la cual se escoge el mejor checkpointing
df_traj=pd.read_csv('experiments_LibrariesRL/results/samplefactory/execution5/process_info/df_traj.csv')
df_traj['traj_rewards']=[np.array(compress_decompress_list(i,compress=False)) for i in list(df_traj['traj_rewards'])]
df_traj['traj_ep_end']=[np.array(compress_decompress_list(i,compress=False)) for i in list(df_traj['traj_ep_end'])]

train_rewards=[rw for policy_trajs in df_traj['traj_rewards'] for traj in policy_trajs for rw in traj]
train_ep_ends=[ep_end for policy_trajs in df_traj['traj_ep_end'] for traj in policy_trajs for ep_end in traj]
n_timesteps_per_iter=list(range(64*10,64*10*8,64*10))
stats_window_size=100

print(training_stats_single_worker(train_rewards,train_ep_ends,n_timesteps_per_iter,stats_window_size))

# Comprobar la metrica para ejecucion en paralelo
total_timesteps=64*10*5
Options.learn_process(method,env,seed,total_timesteps,'execution6',library_dir,
                            n_steps_per_env=64,n_workers=2,n_envs_per_worker=1,
                            batch_size=64*10,n_batches_per_epoch=1,n_epoch=1,
                            save_every_sec=1, keep_checkpoints=8,save_best_every_sec=1,save_best_after=total_timesteps)
external_run('experiments_LibrariesRL/samplefactory.py',list(range(48,56))+list(range(178,184)))

df_traj=pd.read_csv('experiments_LibrariesRL/results/samplefactory/execution6/process_info/df_traj.csv')
df_traj['traj_rewards']=[np.array(compress_decompress_list(i,compress=False)) for i in list(df_traj['traj_rewards'])]
df_traj['traj_ep_end']=[np.array(compress_decompress_list(i,compress=False)) for i in list(df_traj['traj_ep_end'])]
n_timesteps_per_iter=list(range(64*10,64*10*8,64*10))
stats_window_size=100
df_traj_rewards=[[],[]]
df_traj_ep_end=[[],[]]
for i in range(len(df_traj['traj_rewards'])):
    # Worker 1
    df_traj_rewards[0]+=np.array(df_traj['traj_rewards'][i][::2]).flatten().tolist()
    df_traj_ep_end[0]+=np.array(df_traj['traj_ep_end'][i][::2]).flatten().tolist()
    # Worker 2
    df_traj_rewards[1]+=np.array(df_traj['traj_rewards'][i][1::2]).flatten().tolist()
    df_traj_ep_end[1]+=np.array(df_traj['traj_ep_end'][i][1::2]).flatten().tolist()

n_timesteps_per_iter=list(range(64*10,64*10*8,64*10))
stats_window_size=100

print(training_stats_multiple_workers(df_traj_rewards,df_traj_ep_end,n_timesteps_per_iter,stats_window_size))

