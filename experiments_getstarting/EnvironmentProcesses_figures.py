'''
Graficas que ilustran diferentes procesos de un environment hasta convergencia, mostrando diferencias entre procesos en:
ritmo de convergencia, tiempo de convergencia y optimo local alcanzado. Los datos mas relevantes de las graficas para cada
environment se guardan en una tabla. 
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from tqdm import tqdm


def plot_environment_processes(list_seeds,env_name,reward_threshold):

    fig=plt.figure(figsize=[10,4])
    plt.subplots_adjust(left=0.08,bottom=0.27,right=0.97,top=0.84,wspace=0.39,hspace=0.2)

    #----------------------------------------------------------------------------------------------
    # GRAFICA 1: learning-curves (cortadas en n_policy relevantes)
    #----------------------------------------------------------------------------------------------
    ax=plt.subplot(131)
    ax.grid(True, which='both',linestyle='--', linewidth=0.8,alpha=0.2)

    all_x_conv=[]
    all_y_max=[]
    matrix_y=[]

    for seed in tqdm(list_seeds):

        # Leer bases de datos.
        df_test=pd.read_parquet('experiments_getstarting/results/EnvironmentProcesses/'+str(env_name)+'/df_test_'+str(env_name)+'_seed'+str(seed)+'.parquet')
        df_train=pd.read_parquet('experiments_getstarting/results/EnvironmentProcesses/'+str(env_name)+'/df_train_'+str(env_name)+'_seed'+str(seed)+'.parquet')

        # Guardar coordenadas de los puntos de las curvas.
        x=list(df_train['n_train_timesteps'])

        y=[]
        y_max=-np.Inf
        for i in list(df_test['n_policy']):

            new_y_max=df_test[df_test['n_policy']<=i]['mean_reward'].max()
            y.append(new_y_max)

            if reward_threshold is None:
                condition=new_y_max-y_max>0
            else:
                condition=new_y_max-y_max>reward_threshold*0.001


            if condition:
                y_max=new_y_max
                x_conv=i

        all_x_conv.append(x_conv)
        all_y_max.append(y_max)
        matrix_y.append(y)

    # Dibujar learning-curves de cada proceso.
    for y in matrix_y:
        x_lim=int(np.mean(all_x_conv)+0.1*np.mean(all_x_conv))
        plt.plot(x[:x_lim], y[:x_lim], linewidth=1,color='grey')

    n_steps=min(x)
    all_x_conv=[i*n_steps for i in all_x_conv]
    if reward_threshold is not None:
        plt.axhline(y=reward_threshold,color='black', linestyle='--')
    plt.axvline(x=np.mean(all_x_conv),color='red', linestyle='--')
    plt.xlim(0,x_lim*n_steps)

    ax.set_xlabel("Train time steps",fontsize=10)
    ax.set_ylabel("Best test reward",fontsize=10)
    ax.set_title('Learning-curve: Best policy found\n',fontsize=10)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0)) 

    #----------------------------------------------------------------------------------------------
    # GRAFICA 2: train time steps necesarios para alcanzar la convergencia por cada proceso.
    #----------------------------------------------------------------------------------------------
    ax=plt.subplot(132)
    ax.grid(True, axis='y',linestyle='--', linewidth=0.8,alpha=0.2)

    ax.bar(range(1,len(list_seeds)+1),all_x_conv,width=0.7,color='grey')
    plt.axhline(y=np.mean(all_x_conv),color='red', linestyle='--')

    ax.set_xlabel("Process",fontsize=10) 
    ax.set_ylabel("Train time steps to converge",fontsize=10)

    #----------------------------------------------------------------------------------------------
    # GRAFICA 3: optimo al que converge cada proceso.
    #----------------------------------------------------------------------------------------------
    ax=plt.subplot(133)
    ax.grid(True, axis='y',linestyle='--', linewidth=0.8,alpha=0.2)

    ax.bar(range(1,len(list_seeds)+1),all_y_max,width=0.7,color='grey')

    plt.ylim(min(all_y_max)-np.std(all_y_max), max(all_y_max)+np.std(all_y_max))
    ax.set_xlabel("Process",fontsize=10) # \n Reformulated condition 2:\nprocesses converge at\nsignificantly different speeds
    ax.set_ylabel("Test reward of convergence",fontsize=10)

    plt.savefig('experiments_getstarting/results/EnvironmentProcesses/ProcessConvergence_'+str(env_name)+'.pdf')
    plt.show()
    plt.close()

    return all_x_conv, all_y_max

# Dibujar graficas por environment y guardar datos relevantes.
envs_seeds=[["InvertedDoublePendulum-v4",[3,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23]],
            ["Ant-v4",list(range(1,21))],
            ["Humanoid-v4",list(range(1,21))]]
info_table=[]
for env_name, list_seeds in envs_seeds:
    print(env_name)
    env=gym.make(env_name)
    all_x_conv,all_y_max=plot_environment_processes(list_seeds,env_name[:-3],env.spec.reward_threshold)
    info_table.append([env_name,min(all_x_conv),max(all_x_conv),np.mean(all_x_conv),env.spec.reward_threshold,min(all_y_max),max(all_y_max),np.mean(all_y_max)])

info_table=pd.DataFrame(info_table,columns=['env_name','min_timesteps_conv','max_timesteps_conv','mean_timesteps_conv',
                                            'reward_threshold','min_reward_conv','max_reward_conv','mean_reward_conv'])
info_table.to_csv('experiments_getstarting/results/EnvironmentProcesses/info_table.csv')

