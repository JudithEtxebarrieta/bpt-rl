'''
Este script contiene experimentos preliminares para ver si son ciertas mis intuiciones sobre:
Relacion entre algoritmos que convergen a politicas deterministas/estocasticas en entornos 
deterministas/estocasticos con la degradacion.

Intuicion concreta:
El diseño del algoritmo (e.g. maximiza entropia o no), puede dar lugar a aprendizajes que tienden a generar
secuencias de politicas que convergen en politicas mas/menos estocasticas. Al mismo tiempo, los entornos pueden
tener una demanda de estocasticidad para ser mejor resueltos (relacionado con lo variables que pueden ser los
episodios generables desde los posibles estados iniciales). Si las dos cosas anteriores no coinciden, puede que
los procesos de aprendizaje sufran de mayor degradacion.

NOTE: 
Mantengo el script por si me vale en el futuro, pero ahora mismo no funciona, por haber eliminado datos que ya no uso
aunque el codigo depende de ellos. Sobre todo son utiles las graficas que tengo guardadas que ejecute usando este codigo.

Conclusiones a las que llegamos con este analisis de prueba:
- Puede haber una tendencia entre value_loss y la degradacion. Podria ser una metrica a monitorizar para proponer un nuevo criterio.
Se observa una simetria en la evolucion de value_loss y truth.
- Aunque los algoritmos consideren politicas estocasticas, esto se hace por sus beneficios de exploracion durante el aprendizaje. Pero
para validar es normal transformar las politicas estocasticas aprendidas en deterministas. Una vez aprendida no nos interesa la
estocasticidad. 

'''

import os, sys
import pandas as pd
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from CriteriaComparison import Converter, Estimator

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#==================================================================================================
# Estudiar la evolucion de diferentes metricas calculadas durante el aprendizaje con algoritmos
# On-policy (PPO) y Off-policy (SAC)
#==================================================================================================

# Generar datos necesarios (aprovechando funciones del otro script)
def generate_data(algo,env,seed,resources):

    # Leer bases de datos del proceso de interes
    current_path=parent_dir+'/_bender/project_SB3/data/'+algo+'_'+env+'_seed'+str(seed)+'_'+resources
    df_test=pd.read_csv(current_path+'/df_val.csv')
    df_train=pd.read_csv(current_path+'/df_traj.csv')

    # Añadir columna de truth si no existe
    df_test['ep_rewards']=[Converter.compress_decompress_list(i,compress=False) for i in df_test['ep_rewards']]
    df_val_estimates=Estimator.read_create_estimates_csv(current_path+'/df_val_estimates.csv',df_test.shape[0])
    Estimator.read_create_estimates_csv(current_path+'/df_traj_estimates.csv',df_test.shape[0])
    if 'truth' not in df_val_estimates.columns.tolist():
        df_val_estimates['truth']=[ np.mean(i[:500]) for i in df_test['ep_rewards'] ]
        df_val_estimates.to_csv(current_path+'/df_val_estimates.csv', index=False)
    if 'truth_norm' not in df_val_estimates.columns.tolist():
        df_val_estimates['truth_norm']=[ (i-min(df_val_estimates['truth']))/(max(df_val_estimates['truth'])-min(df_val_estimates['truth'])) for i in df_val_estimates['truth'] ]
        df_val_estimates.to_csv(current_path+'/df_val_estimates.csv', index=False)

    # Añadir columna de degradacion si no existe
    # process_id=algo+'_'+env+'_'+'seed'+str(seed)
    # global_deg_metric='best_last_deg'
    # local_deg_metric='paired_diff_probpos'

    # generator=EvolutionGenerator(algo,env,seed,resources,0.1)
    # df_degradation=Estimator.read_create_estimates_csv('experiments_intuition/results/DegradationPatterns/data/level_degradation_offpolicy.csv',df_test.shape[0],generator.start_iter)
    # print(generator.start_iter)
    # print(generator.start_time)
    # print(df_test.shape[0])
    # #print(list(df_train['time_seconds'].iloc[generator.start_iter:]))
    
    # df_degradation[process_id+'_'+global_deg_metric+'_'+local_deg_metric]=[generator.degradation_level(time,global_deg_metric,local_deg_metric) for time in list(df_train['time_seconds'].iloc[generator.start_iter:])]
    # df_degradation.to_csv('experiments_intuition/results/DegradationPatterns/data/level_degradation_offpolicy.csv', index=False)

# Evolucion de diferentes metricas durante el aprendizaje 
def finding_patterns_in_process(algo,env,seed,resources):

    process_id=algo+'_'+env+'_'+'seed'+str(seed)
    global_deg_metric='best_last_deg'
    local_deg_metric='paired_diff_probpos'

    # Cargar base de datos con los datos de interes
    current_path=parent_dir+'/_bender/project_SB3/data/'+algo+'_'+env+'_seed'+str(seed)+'_'+resources
    df_traj=pd.read_csv(current_path+'/df_traj.csv')
    df_val_estimates=pd.read_csv(current_path+'/df_val_estimates.csv')
    #df_deg=pd.read_csv('experiments_intuition/results/DegradationPatterns/data/level_degradation.csv')

    # Listas con datos truth, degradaciones, y algunas metricas almacenadas durante el aprendizaje (convergencia, estimaciones objetivo)
    all_truth_norm=list(df_val_estimates['truth_norm'])

    #all_deg=df_deg[process_id+'_'+global_deg_metric+'_'+local_deg_metric]

    all_ent=list(df_traj['entropy_loss'])
    all_ent_norm=Converter.normalize_list(all_ent)

    #all_kl_div=df_traj['KL_div'].iloc[int(df_traj.shape[0]*0.1):]
    all_value_loss_norm=Converter.normalize_list(df_traj['critic_loss'])
    #all_policy_loss_norm=Converter.normalize_list(df_traj['policy_loss'].iloc[int(df_traj.shape[0]*0.1):])
    all_policy_loss_norm=Converter.normalize_list(df_traj['actor_loss'])


    # Grafica: comparacion de curvas de evolucion
    plt.subplots(figsize=(20, 6))
    x=list(range(len(all_ent_norm)))
    #plt.plot(x,all_truth_norm,label='truth')
    plt.plot(x,all_ent_norm,label='entropy')
    #plt.plot(x,all_kl_div,label='KL divergence')
    plt.plot(x,all_value_loss_norm,label='critic loss')
    #plt.plot(x,all_policy_loss_norm,label='policy loss')
    plt.plot(x,all_policy_loss_norm,label='actor loss')
    plt.title(str(process_id))
    plt.legend()
    plt.savefig('experiments_intuition/results/DegradationPatterns/'+process_id+'_patterns.pdf')
    plt.show()

# Main
#---------- Generar estimaciones de las procesos nuevos
for env in ['Ant']:
    for seed in range(1,5):
        generate_data('SAC',env,seed,'analisys')

#---------- Evolucion de metricas almacenadas durante el aprendizaje
for env in ['Ant','Hopper','Humanoid']:
    for seed in range(1,3):
        finding_patterns_in_process('PPO',env,seed,'analisys')

for env in ['Ant']:
    for seed in range(1,5):
        finding_patterns_in_process('SAC',env,seed,'analisys')


#==================================================================================================
# Estudiar la posible dependencia de la degradacion con la correcta relacion entre tendencia
# de estocasticidad definida por el algoritmo y estocasticidad requerida por el entorno
#==================================================================================================
# Distribucion de degradacion por entorno, dependiendo del algoritmo on-policy/off-policy
def PPO_deg_per_env():
    # Leer base de datos con degradaciones online de los procesos
    df = pd.read_csv('experiments_intuition/results/CriteriaComparison/data/deg_mag/level_degradation.csv')
    #df = pd.read_csv('experiments_intuition/results/DegradationPatterns/data/level_degradation.csv')

    # Quedarnos unicamente con las columnas de la ultima degradacion definida (probabilidad positiva de diferencia pareada)
    algo=['PPO']
    envs=['Ant','Humanoid']
    seeds=[f"seed{i}" for i in range(1, 4)]
    deg=['best_last_deg_paired_diff_probpos']

    columns = ["_".join(comb) for comb in itertools.product(algo, envs, seeds, deg)]
    df = df[columns]

    # Dibujar tres KDE de las degradaciones en Ant ,Humanoid y HumanoidStandup
    plt.figure(figsize=(8, 6))

    colors=['red','green','blue']
    x = np.linspace(0, 1, 200)
    for i in range(len(envs)):
        sample=df.filter(like=envs[i]).values.flatten()
        kde= gaussian_kde(sample)
        plt.fill_between(x, kde(x), alpha=0.4, label=envs[i],color=colors[i])
        plt.axvline(np.median(sample), linestyle="-", linewidth=1,color=colors[i])
    print(sample.shape)

    plt.xlabel("Degradation")
    plt.ylabel("KDE")
    plt.legend()
    #plt.title("PPO converge a politicas deterministas ->\n Con entornos mas estocasticos mas degradacion")
    plt.grid(True, alpha=0.3)

    plt.savefig('experiments_intuition/results/DegradationPatterns/PPO_KDE_per_env_det.pdf')
    plt.show()

    # Con mas detalle
    points=np.linspace(0, 1, 5+1)
    intervals= list(zip(points[:-1], points[1:]))
    x = np.linspace(0, 1, 200)
    colors=['red','green','blue']

    fig, axes = plt.subplots(5,1, figsize=(5,10), sharex=True, sharey=True)
    plt.subplots_adjust(left=0.22,bottom=0.05,right=0.9,top=0.88,wspace=0.2,hspace=0.2)

    for ax, (start, end) in zip(axes, intervals):
        # Convertir a índices de filas
        i_start = int(len(df) * start)
        i_end = int(len(df) * end)
        
        # Subconjunto de datos
        subset = df.iloc[i_start:i_end]
        
        for i in range(len(envs)):
            sample=subset.filter(like=envs[i]).values.flatten()
            kde= gaussian_kde(sample)
            ax.fill_between(x, kde(x), alpha=0.4, label=envs[i],color=colors[i])
            ax.axvline(np.median(sample), linestyle="-", linewidth=1,color=colors[i])
            ax.set_ylabel(f"{int(start*100)}% - {int(end*100)}%\n learning time\n\n KDE")
            ax.grid(True, alpha=0.3)
        print(sample.shape)

    plt.title("", fontsize=16)
    plt.xlabel('Degradation')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    plt.savefig('experiments_intuition/results/DegradationPatterns/PPO_KDE_per_env_time.pdf')
    plt.show()

# Analizando la demanda de estocasticidad de cada entorno (redimiento de politicas malas/buenas deterministas/estocasticas)
def eval_policy(policy,env,n_eval_ep,n_workers=1,deterministic=False):
    
    def eval_single_episode(args):
        env_name,policy,episode=args

        env=gym.make(env_name)
        if not isinstance(env, VecEnv):
            env = DummyVecEnv([lambda: env])
        env.seed(0)

        obs=[env.reset() for _ in range(episode)][-1]# La lista de estados iniciales con interaccion coincide con los estados iniciales impares sin interaccion.

        init_obs=obs
        episode_rewards = 0
        episode_len=0
        entropy_list=[]

        done = False # Parameter that indicates after each action if the episode continues (False) or is finished (True).

        with th.no_grad():
            while not done:
                action, _states = policy.predict(obs, deterministic=deterministic) # The action to be taken with the model is predicted.       
                obs, reward, done, info = env.step(action) # Action is applied in the environment.
                episode_rewards+=reward # The reward is saved.
                episode_len+=1

                dist = policy.get_distribution(policy.obs_to_tensor(obs)[0])
                entropy_list.append(dist.entropy())

        return episode_rewards, episode_len, init_obs, entropy_list

    def parallel_eval(policy,env_name,n_eval_ep,n_workers):
        # Set up the parallel processing pool
        results=Parallel(n_jobs=n_workers, backend="loky")(
                delayed(eval_single_episode)([env_name,policy,episode]) for episode in range(1,n_eval_ep+1))
            
        # Split the results into rewards and episode lengths
        all_episode_reward, all_episode_len, all_init_state,all_entropy_list= zip(*results)

        return [float(i) for i in all_episode_reward], all_episode_len, np.array(all_init_state), np.mean([ent for ent_list in all_entropy_list for ent in ent_list])

    # Evaluar la politica.
    eval_metrics=parallel_eval(policy,env,n_eval_ep,n_workers)
    return eval_metrics

def demand_stochasticity_env(env_name,seed=1,n_eval_ep=100,n_workers=1,policy_dir='directorio donde esta la politica guardada'):

    # Generar politica aleatoria (primera del aprendizaje)
    model = PPO(MlpPolicy,env=env_name,seed=seed)# TODO: mejor cargar la politica de las guardadas en los procesos
    det_ep_rewards, det_episode_lens, det_init_states,_=eval_policy(model.policy,env_name,n_eval_ep,n_workers=n_workers,deterministic=True)
    stoch_ep_rewards, stoch_episode_lens, stoch_init_states,entropy=eval_policy(model.policy,env_name,n_eval_ep,n_workers=n_workers)

    # Grafica de KDE
    kde1 = gaussian_kde(det_ep_rewards)
    kde2 = gaussian_kde(stoch_ep_rewards)

    xmin = min(min(det_ep_rewards), min(stoch_ep_rewards))
    xmax = max(max(det_ep_rewards), max(stoch_ep_rewards))
    x = np.linspace(xmin, xmax, 300)

    plt.figure(figsize=(7,5))
    plt.plot(x, kde1(x), label="Deterministic", linewidth=2)
    plt.plot(x, kde2(x), label="Stochastic "+str(entropy), linewidth=2)
    plt.xlabel("Episodic reward")
    plt.ylabel("KDE")
    plt.title("")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('experiments_intuition/results/DegradationPatterns/env_stoch_'+str(env_name)+'.pdf')
    # plt.show()
    
# Main
#---------- Degradaciones por entorno 
PPO_deg_per_env()

#---------- Demanda de estocasticidad
# demand_stochasticity_env('Hopper-v4')
# demand_stochasticity_env('Ant-v4')
# demand_stochasticity_env('Humanoid-v4')

