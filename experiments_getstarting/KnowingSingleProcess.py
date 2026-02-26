
'''


- InvertedDoublePendulun no parece que tiene multiples optimos locales, y en general en cualquier environmet?
Yo creo que la segunda condicion deberia de ser que diferentes procesos tengan diferentes ritmos de convergencia. El PPO tiene
integrado un mecanismo que si ve que la politica no mejora introduce mas aleatoriedad, por eso, tarde o temprano siempre se
alcanza el mismo optimo.

- Quedarnos con la mejor politica observada en lugar de la ultima implica una ligera mejora (amarilla mejor que negra). Pero 
el tiempo extra de test que hay que invertir hace que esa mejora no componse (resto de curvas a la amarilla peoares que la negra).

- Entonces, podemos obtener la informacion de cual es la mejor politica observada reciclando el train reward?
#############
- Test reward: cuando cambia la mejor politica, y como de importante es el cambio (incremento de test reward)
- Train reward: definir diferentes medidas para representar el train reward obtenido con cada politica (trayectoria),
y segun esa medida, mirar cuando cambia la mejor politica.

Con esos datos, construir una grafica tile. El eje OX son las politicas visitadas por orden de visita. La primera fila (arriba)
sera degradada, blanco= no hay cambio de maximo, negro= hay cambio y es grande. Las demas filas, cada una correspondera a una 
diferente medida usada para resumir el train reward de la interaccion de cada politica, y sera una tile rojo=no hay cambio de maximo
y verde=si hay cambio.

Esta grafica permitira ver que criterio detecta mejor los cambios importantes (los que nos interesan) en test reward. Ademas,
aunque no detecte todos los cambios importantes, podremos ver si detecta los importantes (en los que el cambio de reward es grande).
#################

- Tengo que determinar que episodios usar para validar, yo apuesto por usar los mismos para que las comparaciones tengan sentido. 
En ese caso, para dibujar cada learning-curve habria que hacer un probedio, ya que cuando se diga que la validacion se hace con 1,2,3,..
episodios aunque siempre son los mismos 1,2 y 3, las formas de escogerlos pueden ser diferentes (estado inicial de semilla).

-Con Ant empiezan a verse resultados mas interesantes, porque validar con pocos episodios da lugar a malas selecciones de politica
optima. Se observa que es necesario buscar un equilibrio entre frecuencia de validacion y precision de validacion. Por eso, aplicar
el criterio de la semana pasada puede ser interesante en este caso, quitando el umbral.

-Me doy cuenta que cuantos mas steps/ politicas haya que entrenar para converger, el calculo de las metricas a partir del train
sera mas costoso. Por eso, las matrizes que se calculan de rewards de validacion por politica para diferentes metricas a partir de 
train, habria que guardarlos en una base de datos para no tener que calcularlos cada vez que dibujo una grafica.

'''
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
from matplotlib.colors import PowerNorm
from tqdm import tqdm

from our_library import  UtilsDataFrame, UtilsFigure

#==================================================================================================
# Funciones auxiliares
#==================================================================================================
def max_change_in_list(value_list):
    max_change_bool=[]
    change_values=[]
    max_value=value_list[0]
    for i in value_list:
        if len(max_change_bool)>0 and i>max_value:
            max_change_bool.append(True)
            change_values.append(i-max_value)
            max_value=i
        else:
            max_change_bool.append(False)
            change_values.append(0)

    # Normalizar
    change_values=[(i-min(change_values))/(max(change_values)-min(change_values))  if i!= 0 else i for i in change_values]

    return max_change_bool,change_values

def train_reward_metrics_per_policy(matrix_train_rewards):
    metrics_mean=[]
    metrics_max=[]
    metrics_percentile=[]

    for i in matrix_train_rewards:
        metrics_mean.append(np.mean(i))
        metrics_max.append(max(i))
        metrics_percentile.append(np.percentile(i,0.75))

    return metrics_mean,metrics_max,metrics_percentile

def ranking_from_argsort(argsort):
    ranking=[0]*len(argsort)
    ranking_pos=1
    for i in argsort:
        ranking[i]=ranking_pos
        ranking_pos+=1
    return ranking

def rank_labels_from_argsort(labels,argsort):
    new_labels=[]
    for i in argsort:
        new_labels.append(labels[i])
    return new_labels

def form_df_train_to_per_policy_ep_train_rewards(env_name,seed,max_train_timesteps):

    df_train=pd.read_parquet('experiments_getstarting/results/EnvironmentProcesses/'+str(env_name)+'/df_train_'+str(env_name)+'_seed'+str(seed)+'.parquet')
    df_train['train_rewards']=[UtilsDataFrame.compress_decompress_list(i,compress=False) for i in list(df_train['train_rewards'])]
    df_train['train_ep_end']=[UtilsDataFrame.compress_decompress_list(i,compress=False) for i in list(df_train['train_ep_end'])]

    # Obtener listas con rewards train por step e inicio de cada episodio train.
    df_train=df_train[df_train['n_train_timesteps']<=max_train_timesteps]
    n_policies=list(df_train['n_policy'])

    train_rewards=[]
    train_ep_end=[]
    for i in n_policies:
        train_rewards+=list(df_train[df_train['n_policy']==i]['train_rewards'])[0]
        train_ep_end+=list(df_train[df_train['n_policy']==i]['train_ep_end'])[0]

    n_train_timesteps=list(df_train['n_train_timesteps'])

    # Obtener lista con el numero de episocios train evaluados por cada politica durante el entrenamiento.
    num_ep_start_policy=[]
    ep_rew_mean=[]
    last_i=0# Last episode end
    current_i=0# Current step

    for i in train_ep_end:
        if i:
            ep_rew_mean.append(sum(train_rewards[last_i:current_i]))
            last_i=current_i
        current_i+=1

        if current_i in n_train_timesteps:
            num_ep_start_policy.append(len(ep_rew_mean))

    # Obtener matriz con los train rewards por episodio asociados a cada politica por fila.
    policy_ep_reward_matrix=[]
    for i in range(len(num_ep_start_policy)):
        if i==0:
            policy_ep_reward_matrix.append(ep_rew_mean[:num_ep_start_policy[i]])
        else:
            policy_ep_reward_matrix.append(ep_rew_mean[num_ep_start_policy[i-1]:num_ep_start_policy[i]])
     

    return policy_ep_reward_matrix

def form_df_train_to_matrix_train_rewards(list_window_sizes,env_name,seed,max_train_timesteps,batch_size=None):

    df_train=pd.read_parquet('experiments_getstarting/results/EnvironmentProcesses/'+str(env_name)+'/df_train_'+str(env_name)+'_seed'+str(seed)+'.parquet')
    df_train['train_rewards']=[UtilsDataFrame.compress_decompress_list(i,compress=False) for i in list(df_train['train_rewards'])]
    df_train['train_ep_end']=[UtilsDataFrame.compress_decompress_list(i,compress=False) for i in list(df_train['train_ep_end'])]


    df_train=df_train[df_train['n_train_timesteps']<=max_train_timesteps]
    n_policies=list(df_train['n_policy'])

    train_rewards=[]
    train_ep_end=[]
    for i in n_policies:
        train_rewards+=list(df_train[df_train['n_policy']==i]['train_rewards'])[0]
        train_ep_end+=list(df_train[df_train['n_policy']==i]['train_ep_end'])[0]

    n_train_timesteps=list(df_train['n_train_timesteps'])
    reward_matrix=[]

    # Añadido para guardar por batches de iteraciones los rewards de los episodes de train
    batch_ep_reward_matrix=[]
    num_ep_start_batch=[]
    first_iteration=True
    ##
    
    for stats_window_size in tqdm(list_window_sizes):
        rewards=[]

        ep_rew_mean=[]
        last_i=0# Last episode end
        current_i=0# Current step

        for i in train_ep_end:
            if i:
                ep_rew_mean.append(sum(train_rewards[last_i:current_i]))
                last_i=current_i
            current_i+=1

            if current_i in n_train_timesteps:
                
                if len(ep_rew_mean)<stats_window_size:
                    rewards.append(np.mean(ep_rew_mean))
                else:
                    rewards.append(np.mean(ep_rew_mean[-stats_window_size:]))

                # Añadido para guardar por batches de iteraciones los rewards de los episodes de train
                if batch_size is not None and first_iteration:
                    if (current_i//min(n_train_timesteps))%batch_size==0:
                        num_ep_start_batch.append(len(ep_rew_mean))
                ##
        first_iteration=False
                    
        reward_matrix.append(rewards)

     # Añadido para guardar por batches de iteraciones los rewards de los episodes de train
    for i in range(len(num_ep_start_batch)):
        if i==0:
            batch_ep_reward_matrix.append(ep_rew_mean[:num_ep_start_batch[i]])
        else:
            batch_ep_reward_matrix.append(ep_rew_mean[num_ep_start_batch[i-1]:num_ep_start_batch[i]])
    ##

    if batch_size is None: 
        return reward_matrix
    else:       

        return reward_matrix,batch_ep_reward_matrix

# Para no tener que calcularlo cada vez que se dibuja una grafica
def extract_validation_data_from_dfs(env_name,seed,max_train_timesteps):
    global df_test, df_train

    # Cargar base de datos
    df_train=pd.read_parquet('experiments_getstarting/results/EnvironmentProcesses/'+str(env_name)+'/df_train_'+str(env_name)+'_seed'+str(seed)+'.parquet')
    n_steps=int(df_train['n_train_timesteps'].min())
    df_test=pd.read_parquet('experiments_getstarting/results/EnvironmentProcesses/'+str(env_name)+'/df_test_'+str(env_name)+'_seed'+str(seed)+'.parquet')
    df_test=df_test[df_test['n_policy']<=max_train_timesteps//n_steps]

    # Crear y guardar bases de datos con informacion extraida del train
    

    window_train_rewards,batch_window_train_rewards=form_df_train_to_matrix_train_rewards([100,50,20,10,5,1],env_name,seed,max_train_timesteps,(max_train_timesteps//n_steps)//9)
    trajec_train_rewards=form_df_train_to_per_policy_ep_train_rewards(env_name,seed,max_train_timesteps)

    if not os.path.exists('experiments_getstarting/results/KnowingSingleProcess/'+str(env_name)+'/extracted_data'):
        os.makedirs('experiments_getstarting/results/KnowingSingleProcess/'+str(env_name)+'/extracted_data')
                
    pickle.dump(window_train_rewards, open('experiments_getstarting/results/KnowingSingleProcess/'+str(env_name)+'/extracted_data/window_train_rewards'+str(seed)+'.pkl', 'wb'))
    pickle.dump(batch_window_train_rewards, open('experiments_getstarting/results/KnowingSingleProcess/'+str(env_name)+'/extracted_data/batch_window_train_rewards'+str(seed)+'.pkl', 'wb'))
    pickle.dump(trajec_train_rewards, open('experiments_getstarting/results/KnowingSingleProcess/'+str(env_name)+'/extracted_data/trajec_train_rewards'+str(seed)+'.pkl', 'wb'))

    # Construir matriz con cambios en maximo rewards considerando diferentes metricas

    # Test reward por politica con maximo n_test_ep
    test_rewards=list(df_test['mean_reward'])
    max_test_reward_change,test_reward_changes=max_change_in_list(test_rewards)

    # Calcular vectores normalizados de cambios en el maximo reward usando las diferentes metricas de validacion
    metrics_mean,metrics_max,metrics_percentile=train_reward_metrics_per_policy(trajec_train_rewards)
    _,metrics_mean_changes=max_change_in_list(metrics_mean)
    _,metrics_max_changes=max_change_in_list(metrics_max)

    window_train_rewaerds_changes=[]
    for i in window_train_rewards:
        _,changes=max_change_in_list(i)
        window_train_rewaerds_changes.append(changes)

    metric_changes_norm=np.array([test_reward_changes,metrics_mean_changes,metrics_max_changes]+window_train_rewaerds_changes)
    pickle.dump(metric_changes_norm, open('experiments_getstarting/results/KnowingSingleProcess/'+str(env_name)+'/extracted_data/metric_changes_norm'+str(seed)+'.pkl', 'wb'))

# Para fijar los episodios que se usaran en la validacion y que las comparaciones sean justas
def validation_ep_selection(ep_test,n_test_ep,test_type='reward'):
    list_output=[]
    list_seeds=range(len(ep_test))
    for i in list_seeds:
        random.seed(i)
        if test_type=='reward':
            list_output.append(np.mean(random.sample(ep_test,n_test_ep)))
        if test_type=='len':
            list_output.append(sum(random.sample(ep_test,n_test_ep)))
        

    return list_output


#==================================================================================================
# GRAFICA 1: validar todas las politicas durante el entrenamiento con numero de episodios test constante
#==================================================================================================
'''
Hay que eliminar ciertos comentarios de esta funcion, ya que es codigo comentado de versiones anteriores.

'''
def learning_curves_test_reward(x,list_n_test_ep,env_name,seed,list_eval_freq=[1]):

    # Para guardar los resultados
    output_path='experiments_getstarting/results/KnowingSingleProcess/'+str(env_name)+'/learning_curves_test_reward'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    eval_freq_with_train=False

    df_test=pd.read_parquet('experiments_getstarting/results/EnvironmentProcesses/'+str(env_name)+'/df_test_'+str(env_name)+'_seed'+str(seed)+'.parquet')
    df_train=pd.read_parquet('experiments_getstarting/results/EnvironmentProcesses/'+str(env_name)+'/df_train_'+str(env_name)+'_seed'+str(seed)+'.parquet')
    df_test=df_test[df_test['n_policy']<=max(x)//2048]
    df_train=df_train[df_train['n_policy']<=max(x)//2048-1]

    n_policies=list(df_test['n_policy'])


    # Dibujar curvas
    fig=plt.figure(figsize=[10,5])
    plt.subplots_adjust(left=0.09,bottom=0.152,right=0.7,top=0.83,wspace=0.39,hspace=0.2)

    ax=plt.subplot(111)
    ax.grid(True, which='both',linestyle='--', linewidth=0.8,alpha=0.2)

    # Dibujar curva sin test durante el proceso (ultima politica observada)
    y=[]
    for timesteps in x:
            last_policy=df_train[df_train['n_train_timesteps']<=timesteps]['n_policy'].max()+1
            y.append(df_test[df_test['n_policy']==last_policy]['mean_reward'])
    plt.plot(x, y, linewidth=1,color='black',label='Without validation (last visited policy)')

    # Dibujar curva con mejor test pero sin contar el extra de tiempo
    y=[]
    for timesteps in x:
            last_policy=df_train[df_train['n_train_timesteps']<=timesteps]['n_policy'].max()+1
            y.append(df_test[df_test['n_policy']<=last_policy]['mean_reward'].max())
    plt.plot(x, y, linewidth=1,color='yellow',label='freq=1  acc=100 (counted as 0)')

    # Resto de curvas
    if len(list_n_test_ep)>1:
        for_list=list_n_test_ep
    else:
        for_list=list_eval_freq
    
    for i in tqdm(for_list):

        
        if len(list_n_test_ep)>1:
            n_test_ep=i
            eval_freq=list_eval_freq[0]
            label=n_test_ep
        else:
            eval_freq=i
            n_test_ep=list_n_test_ep[0]
            label=eval_freq


        current_mean_reward=[]
        current_n_test_timesteps=[]
        cumulative_n_test_timesteps=[0]*100


        for policy in n_policies:
            
            if type(eval_freq) is int:
                condition=policy%eval_freq==0
            else:
                # label,change,threshold=eval_freq
                # condition=change[policy-1]>threshold
                # eval_freq_with_train=True

                label,change=eval_freq
                condition=change[policy-1]>0
                # n_test_ep=int(np.ceil(100*(1-change[policy-1])))# cuanto mayor sea el cambio en el maximo menor precision podemos usar

                eval_freq_with_train=True



            if condition:
                ep_test_rewards_compressed=list(df_test[df_test['n_policy']==policy]['ep_test_rewards'])
                ep_test_len_compressed=list(df_test[df_test['n_policy']==policy]['ep_test_len'])

                ep_test_rewards=UtilsDataFrame.compress_decompress_list(ep_test_rewards_compressed[0],compress=False)
                ep_test_len=UtilsDataFrame.compress_decompress_list(ep_test_len_compressed[0],compress=False)

                ############### Modificando forma en que se escogen los episodios de validacion
                #current_mean_reward.append(np.mean([np.mean(random.sample(ep_test_rewards,n_test_ep)) for i in range(100)]))
                # current_mean_reward.append(np.mean(random.sample(ep_test_rewards,n_test_ep)))
                # current_mean_reward.append(np.mean(ep_test_rewards[:n_test_ep]))

                current_mean_reward.append(validation_ep_selection(ep_test_rewards,n_test_ep))
                

                # cumulative_n_test_timesteps+=sum(random.sample(ep_test_len,n_test_ep))

                for_cumulative_n_test_timesteps=validation_ep_selection(ep_test_len,n_test_ep,test_type='len')
                cumulative_n_test_timesteps=[cumulative_n_test_timesteps[i]+for_cumulative_n_test_timesteps[i] for i in range(100)]

            else:
                #current_mean_reward.append(0)
                current_mean_reward.append([-np.Inf]*100)
            current_n_test_timesteps.append(cumulative_n_test_timesteps)

        #df_test['current_mean_reward']=current_mean_reward
        current_mean_reward=np.array(current_mean_reward).T
        # df_test['current_n_test_timesteps']=current_n_test_timesteps

        # total_timesteps=np.array(df_test['current_n_test_timesteps'])+np.array(df_train['n_train_timesteps'])
        # df_test['total_timesteps']=total_timesteps
        current_n_test_timesteps=np.array(current_n_test_timesteps).T
        total_timesteps=[ i+np.array(df_train['n_train_timesteps']) for i in current_n_test_timesteps]
        min_total_timesteps=max([min(i) for i in total_timesteps])

        y=[]
        x_plot=[]
        for timesteps in x:
            if min_total_timesteps<timesteps:
                # indx_max=df_test[df_test['total_timesteps']<=timesteps]['current_mean_reward'].idxmax()
                # y.append(df_test['mean_reward'][indx_max])
                # x_plot.append(timesteps)

                # indx_max=int(df_test[df_test['total_timesteps']<=timesteps]['n_policy'].max())
                # y_mean,y_q05,y_q95=UtilsFigure.bootstrap_mean_and_confidence_interval([df_test['mean_reward'][np.argmax(i[:indx_max])] for i in current_mean_reward])
                # y.append([y_mean,y_q05,y_q95])
                # x_plot.append(timesteps)


                indx_max=[list(i<=timesteps).index(False) for i in total_timesteps]
                list_argmax=[np.argmax(current_mean_reward[i][:indx_max[i]])for i in range(100)]
                y_mean,y_q05,y_q95=UtilsFigure.bootstrap_mean_and_confidence_interval([df_test['mean_reward'][i] for i in list_argmax])
                y.append([y_mean,y_q05,y_q95])
                x_plot.append(timesteps)

        ax.fill_between(x_plot,np.array(y)[:,1],np.array(y)[:,2], alpha=.2, linewidth=0)
        plt.plot(x_plot, np.array(y)[:,0], linewidth=1,label=label)

    ax.legend(title="",fontsize=8,bbox_to_anchor=(1, 1, 0, 0))

    ax.set_xlabel("Total steps (train+test)\n \n(2048 train steps = training 1 policy = 1 trajectory)",fontsize=10)
    ax.set_ylabel("Test reward (max acc) of the best policy\n(in terms of validation reward)",fontsize=10)

    if len(list_n_test_ep)>1:
        ax.set_title('Learning-curves with\n\n Same validation freq: '+str(eval_freq)+' (policies)\nDifferent validation acc (test episodes, see legend)',fontsize=10)
        plt.savefig(output_path+'/DIFFacc_SAMEfreq'+str(eval_freq)+'_'+str(seed)+'.pdf')
    else:
        ax.set_title('Learning-curves with\n\n Same validation acc: '+str(n_test_ep)+' (test episodes)\nDifferent validation freq (policies, see legend)' ,fontsize=10)
        if eval_freq_with_train:
            plt.savefig(output_path+'/DIFFfreq_withtrain_SAMEacc'+str(n_test_ep)+'_'+str(seed)+'.pdf')
        else:
            plt.savefig(output_path+'/DIFFfreq_SAMEacc'+str(n_test_ep)+'_'+str(seed)+'.pdf')



    plt.show()
    plt.close()


#==================================================================================================
# GRAFICA 2: validar todas las politicas durante el entrenamiento con numero de episodios train constante
#==================================================================================================
def learning_curves_train_reward(x,env_name,seed):

    # Para guardar los resultados
    output_path='experiments_getstarting/results/KnowingSingleProcess/'+str(env_name)+'/learning_curves_train_reward'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    fig=plt.figure(figsize=[10,5])
    plt.subplots_adjust(left=0.09,bottom=0.152,right=0.73,top=0.9,wspace=0.39,hspace=0.2)

    #----------------------------------------------------------------------------------------------
    # GRAFICA 1: learning-curves
    #----------------------------------------------------------------------------------------------
    ax=plt.subplot(111)
    ax.grid(True, which='both',linestyle='--', linewidth=0.8,alpha=0.2)

    #train_reward_matrix=form_df_train_to_matrix_train_rewards(list_window_sizes,env_name,seed,max(x))
    
    train_reward_matrix = pickle.load(open('experiments_getstarting/results/KnowingSingleProcess/'+str(env_name)+'/extracted_data/window_train_rewards'+str(seed)+'.pkl', 'rb'))

    trajec_train_rewards=pickle.load(open('experiments_getstarting/results/KnowingSingleProcess/'+str(env_name)+'/extracted_data/trajec_train_rewards'+str(seed)+'.pkl', 'rb'))

    df_test=pd.read_parquet('experiments_getstarting/results/EnvironmentProcesses/'+str(env_name)+'/df_test_'+str(env_name)+'_seed'+str(seed)+'.parquet')
    df_train=pd.read_parquet('experiments_getstarting/results/EnvironmentProcesses/'+str(env_name)+'/df_train_'+str(env_name)+'_seed'+str(seed)+'.parquet')
    list_window_sizes=[100,50,20,10,5,1]

    y_matrix=[]
    labels=[]
    colors=[]
    default_colors=list(mcolors.TABLEAU_COLORS.keys())

    # Dibujar curva sin test durante el proceso (ultima politica observada)
    y=[]
    for timesteps in x:
            last_policy=df_train[df_train['n_train_timesteps']<=timesteps]['n_policy'].max()+1
            y.append(float(df_test[df_test['n_policy']==last_policy]['mean_reward']))
    plt.plot(x, y, linewidth=1,color='black',label='None (last visited policy)')
    y_matrix.append(y)
    labels.append('None (last visited policy)')
    colors.append('black')

    # Curvas usando trajectorias
    metrics_mean,metrics_max,metrics_percentile=train_reward_metrics_per_policy(trajec_train_rewards)

    y=[]
    for train_timesteps in x:
        last_policy=df_train[df_train['n_train_timesteps']<=train_timesteps]['n_policy'].max()
        best_policy=metrics_mean.index(max(metrics_mean[:last_policy]))
        y.append(float(df_test[df_test['n_policy']==best_policy+1]['mean_reward']))

    plt.plot(x, y, linewidth=1,label='Trajec mean')
    y_matrix.append(y)
    labels.append('Trajec mean')
    colors.append(default_colors[0])

    y=[]
    for train_timesteps in x:
        last_policy=df_train[df_train['n_train_timesteps']<=train_timesteps]['n_policy'].max()
        best_policy=metrics_max.index(max(metrics_max[:last_policy]))
        y.append(float(df_test[df_test['n_policy']==best_policy+1]['mean_reward']))

    plt.plot(x, y, linewidth=1,label='Trajec max')
    y_matrix.append(y)
    labels.append('Trajec max')
    colors.append(default_colors[1])

    # Resto de curvas usando window
    for i in range(len(list_window_sizes)):
        y=[]
        for train_timesteps in x:
            last_policy=df_train[df_train['n_train_timesteps']<=train_timesteps]['n_policy'].max()
            best_policy=train_reward_matrix[i].index(max(train_reward_matrix[i][:last_policy]))
            y.append(float(df_test[df_test['n_policy']==best_policy+1]['mean_reward']))

        plt.plot(x, y, linewidth=1,label='Window '+str(list_window_sizes[i]))
        y_matrix.append(y)
        labels.append('Window '+str(list_window_sizes[i]))
        colors.append(default_colors[i+2])

    ax.legend(title="metric",fontsize=8,bbox_to_anchor=(1.3, 1, 0, 0))
    ax.set_xlabel("Total train steps\n(no extra consumption of test steps in validation here)",fontsize=10)
    ax.set_ylabel("Test reward (max acc) of the best policy\n(in terms of validation metric)",fontsize=10)
    ax.set_title('Learning-curves with\ndifferent metrics from train rewards to validate all policies',fontsize=10)

    plt.savefig(output_path+'/learning_curves'+str(seed)+'.pdf')
    plt.show()
    plt.close()

    #----------------------------------------------------------------------------------------------
    # GRAFICA 2: rankings de las learning-curves durante el entrenamiento
    #----------------------------------------------------------------------------------------------
    fig=plt.figure(figsize=[15,4])
    plt.subplots_adjust(left=0.05,bottom=0.3,right=0.97,top=0.9,wspace=0.39,hspace=0.35)

    ax=plt.subplot(111)
    y_matrix=np.array(y_matrix)
    y_matrix=y_matrix.T
    data=[]
    for i in y_matrix:
        data.append(rank_labels_from_argsort(labels,np.argsort(-np.array(i))))

    data=np.array(data)
    data= data.T

    # Convertir los valores categóricos a números para visualización
    category_map = {k: v for v, k in enumerate(labels)}
    numerical_data = np.vectorize(category_map.get)(data)

    # Dibujar el mapa de calor con color bar discreta
    sns.heatmap(numerical_data, cmap=colors, cbar=True, linewidths=0.5, cbar_kws={"ticks": range(len(labels))})

    # Ajustar la barra de color con las categorías
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks(range(len(labels)))
    colorbar.set_ticklabels(labels)
    colorbar.set_label('metric')

    ax.set_xlabel('Train timesteps')
    ax.set_xticks(range(0,len(x),10))
    ax.set_xticklabels(range(min(x),max(x),int((max(x)-min(x))/10)))
    ax.set_ylabel('From best (top)\nto worts (bottom)')
    ax.set_title('The best metric during training')
    ax.set_yticklabels([],rotation=0)

    plt.savefig(output_path+'/learning_curve_rankings'+str(seed)+'.pdf')
    plt.show()
    plt.close()

#==================================================================================================
# GRAFICA 3: entendiendo relacion entre train y test reward usando rankings
#==================================================================================================
'''
esta funcion solo esta ejecutada para InvertedDoublePendulum. tengo que generalizarla para environments
cuyo tiempo de convergencia sea mayor, y por tanto necesitemos hacer mas de una grafica por batches.
Tengo que usar el codigo de "comparison_test_train_rewards" para la adaptacion.
'''
def comparison_test_train_rewards_rankings(list_window_sizes,env_name,seed,max_train_timesteps):

    fig=plt.figure(figsize=[10,3])
    plt.subplots_adjust(left=0.14,bottom=0.152,right=0.97,top=0.9,wspace=0.39,hspace=0.2)

    #----------------------------------------------------------------------------------------------
    # GRAFICA 1: comparacion de ranking de todas las politicas visitadas durante el entrenamiento,
    # definiendo los ranking a partir de las diferentes formas de validar las politicas (ya sea
    # usando el maximo numero de episodios test, como usando diferentes tamaños de ventana train).
    #----------------------------------------------------------------------------------------------
    ax=plt.subplot(111)
    ax.grid(True, which='both',linestyle='--', linewidth=0.8,alpha=0.2)

    df_train=pd.read_parquet('experiments_getstarting/results/EnvironmentProcesses/'+str(env_name)+'/df_train_'+str(env_name)+'_seed'+str(seed)+'.parquet')
    n_steps=int(df_train['n_train_timesteps'].min())

    df_test=pd.read_parquet('experiments_getstarting/results/EnvironmentProcesses/'+str(env_name)+'/df_test_'+str(env_name)+'_seed'+str(seed)+'.parquet')
    df_test=df_test[df_test['n_policy']<=max_train_timesteps//n_steps]

    ranking_matrix=[]
    y=[]

    # Test reward por politica con maximo n_test_ep
    ranking_matrix.append(ranking_from_argsort(np.argsort(list(df_test['mean_reward']))[::-1]))
    y.append('100 (test)')

    # Train reward por politica con diferentes n_train_ep constantes
    batch_size=25# para la siguiente grafica
    matrix_train_rewards,batch_ep_reward_matrix=form_df_train_to_matrix_train_rewards(list_window_sizes,env_name,seed,max_train_timesteps,batch_size)

    for i in range(len(matrix_train_rewards)):
        ranking_matrix.append(ranking_from_argsort(np.argsort(matrix_train_rewards[i])[::-1]))
        y.append(list_window_sizes[i])

    ranking_matrix=np.matrix(ranking_matrix)

    color = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=True)
    color=cm.get_cmap(color)
    color=color(np.linspace(0,1,ranking_matrix.shape[1]))
    color[:1, :]=np.array([14/256, 241/256, 249/256, 1])# Rojo (codigo rgb)
    color = ListedColormap(color)

    ax = sns.heatmap(ranking_matrix, cmap=color,linewidths=.5, linecolor='lightgray')

    colorbar=ax.collections[0].colorbar
    colorbar.set_label('Ranking position')
    colorbar.set_ticks(range(0,ranking_matrix.shape[1],25))
    colorbar.set_ticklabels(range(1,ranking_matrix.shape[1]+1,25))

    ax.set_xlabel('n_policy')
    ax.set_xticks(range(0,ranking_matrix.shape[1],10))
    ax.set_xticklabels(range(1,ranking_matrix.shape[1]+1,10),rotation=0)
    ax.set_ylabel('n_ep')
    ax.set_title('Comparing rankings depending on accuracy')
    ax.set_yticks(np.arange(0.5,len(y)+0.5,1))
    ax.set_yticklabels(y,rotation=0)

    plt.savefig('experiments_getstarting/results/KnowingSingleProcess/'+str(env_name)+'/comparison_test_train_reward_overall_ranking'+str(seed)+'.pdf')
    plt.show()
    plt.close()

    #----------------------------------------------------------------------------------------------
    # GRAFICA 2: misma grafica anterior pero dividida en batches de politicas, para ver si 
    # el ranking mas parecido al primero se da con diferenetes tamaños de ventana en diferentes
    # etapas del entrenamiento. Además se añaden por batches los trozos correspondientes a 
    # la learning-curve por defecto y la curva de train reward por episodio train recolectado.
    #----------------------------------------------------------------------------------------------
    fig=plt.figure(figsize=[15,5])
    plt.subplots_adjust(left=0.08,bottom=0.152,right=0.96,top=0.9,wspace=0.16,hspace=0.2)

    def ranking_error(ranking,best_argsort):
        ranking_error=[]
        for i in best_argsort:
            ranking_error.append(abs((i+1)-ranking[i]))
        return ranking_error
    
    def reorder_matrix_by_first_row(matrix):
        order=np.argsort(matrix[0])
        new_matrix=[]
        for i in matrix:
            new_matrix.append(rank_labels_from_argsort(i,order))
        return new_matrix

    matrix_train_rewards=np.array(matrix_train_rewards)


    # LINEA 3: comparacion de errores de ordenacion de politicas por batch
    #----------------------------------------------------------------------------------------------
    n_batch=1
    matrix_batch_mean_rewards=[]
    for i in range(1,df_test.shape[0]+1):
        if i%batch_size==0:

            ranking_matrix=[]
            y=[]

            # Test reward por politica con maximo n_test_ep
            mean_rewards=list(df_test.iloc[:i]['mean_reward'])
            ranking_matrix.append(ranking_from_argsort(np.argsort(mean_rewards[-batch_size:])[::-1]))
            y.append('100 (test)')


            # Train reward por politica con diferentes n_train_ep constantes
            batch_matrix_train_rewards=matrix_train_rewards[:,(i-batch_size+1):(i+1)]
            for i in range(len(batch_matrix_train_rewards)):
                ranking_matrix.append(ranking_from_argsort(np.argsort(batch_matrix_train_rewards[i])[::-1]))
                y.append(list_window_sizes[i])

            # Reordenar matriz de acuerdo a orden de primera fila
            ranking_matrix=reorder_matrix_by_first_row(ranking_matrix)

            # Definir matriz de error de posiciones
            pos_err_matrix=[]
            best_argsort=np.argsort(ranking_matrix[0])

            for ranking in ranking_matrix:
                new_ranking_err=ranking_error(ranking,best_argsort)
                norm_ranking_err=[i/max(new_ranking_err) for i in new_ranking_err]
                pos_err_matrix.append(norm_ranking_err)
                
            # Para dibujar despues los trozos de learning-curve asociados a las politicas de cada batch
            mean_rewards=list(mean_rewards[-batch_size:])
            matrix_batch_mean_rewards.append(mean_rewards)
            
            # Dibujar ahora
            ax=plt.subplot(3,df_test.shape[0]//batch_size,2*(df_test.shape[0]//batch_size)+n_batch)
            pos_err_matrix=np.matrix(pos_err_matrix)

            color = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=False, as_cmap=True)

            if n_batch==df_test.shape[0]//batch_size:
                ax = sns.heatmap(pos_err_matrix, cmap=color,linewidths=.5, linecolor='lightgray')
                colorbar=ax.collections[0].colorbar
                colorbar.set_label('Normaliced ranking\n position error')
                ax.set_yticklabels([])


            elif n_batch==1:
                ax = sns.heatmap(pos_err_matrix, cmap=color,linewidths=.5, linecolor='lightgray',cbar=False)
                ax.set_ylabel('n_train_ep')
                ax.set_yticks(np.arange(0.5,len(y)+0.5,1))
                ax.set_yticklabels(y,rotation=0)
            
            else:
                ax = sns.heatmap(pos_err_matrix, cmap=color,linewidths=.5, linecolor='lightgray',cbar=False)
                ax.set_yticklabels([])

            ax.set_xlabel('Batch'+str(n_batch)+'\nof '+str(batch_size)+'policies')
            ax.set_xticks(range(0,pos_err_matrix.shape[1],10))
            ax.set_xticklabels(range(batch_size*(n_batch-1),batch_size*n_batch,10),rotation=0)
            ax.set_xticklabels([])
            

            n_batch+=1

    # LINEA 2: trozos de learning-curve por defecto (test reward de ultima politica visitada) por batch
    #----------------------------------------------------------------------------------------------
    matrix_batch_mean_rewards=np.array(matrix_batch_mean_rewards)
    matrix_batch_mean_rewards=matrix_batch_mean_rewards/np.max(matrix_batch_mean_rewards)
    for i in range(1,n_batch):
        ax=plt.subplot(3,df_test.shape[0]//batch_size,df_test.shape[0]//batch_size+i)
        plt.plot(range(batch_size),matrix_batch_mean_rewards[i-1],color='grey')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_ylim(-0.1,1.1)
        if i==1:
            ax.set_ylabel('Learning-curve\nlast visited policy\n(normalized)')

    # LINEA 1: trozo de "curva de rewards por episodio de entrenamiento" por batch
    #----------------------------------------------------------------------------------------------
    max_value=-np.Inf
    for i in batch_ep_reward_matrix:
        max_row=max(i)
        if max_row>max_value:
            max_value=max_row
    norm_batch_ep_reward_matrix=[]
    for i in batch_ep_reward_matrix:
        norm_batch_ep_reward_matrix.append(np.array(i)/max_value)

    for i in range(1,n_batch):
        ax=plt.subplot(3,df_test.shape[0]//batch_size,i)
        plt.plot(range(len(norm_batch_ep_reward_matrix[i-1])),norm_batch_ep_reward_matrix[i-1],color='grey')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_ylim(-0.1,1.1)
        if i==1:
            ax.set_ylabel('Train rewards\nper episode\n(normalized)')

    plt.savefig('experiments_getstarting/results/KnowingSingleProcess/'+str(env_name)+'/comparison_test_train_reward_per_batches'+str(seed)+'.pdf')
    plt.show()
    plt.close()

    #----------------------------------------------------------------------------------------------
    # GRAFICA 3: 
    #----------------------------------------------------------------------------------------------

    # Test reward por politica con maximo n_test_ep
    test_rewards=list(df_test['mean_reward'])
    max_test_reward_change,_=max_change_in_list(test_rewards)

    # Train reward por episodio con cada politica (basado en trayectorias propias)
    matrix_train_rewards=form_df_train_to_per_policy_ep_train_rewards(env_name,seed,max_train_timesteps)

    # Representar curva por batch
    fig=plt.figure(figsize=[10,10])
    plt.subplots_adjust(left=0.08,bottom=0.07,right=0.96,top=0.9,wspace=0.16,hspace=0.55)

    batch_size=25
    n_batches=0
    for i in range(len(test_rewards)):
        if (i+1)%batch_size==0:

            ax=plt.subplot(df_test.shape[0]//batch_size,1,n_batches+1)

            # Curva de test rewards
            x=range(n_batches*batch_size*n_steps+1,(n_batches+1)*batch_size*n_steps,n_steps)
            y=test_rewards[(n_batches*batch_size):((n_batches+1)*batch_size)]
            max_change=max_test_reward_change[(n_batches*batch_size):((n_batches+1)*batch_size)]

            
            change_True=[i for i in range(len(max_change)) if max_change[i]]
            change_False=[i for i in range(len(max_change)) if not max_change[i]]
    
            plt.scatter([x[i] for i in change_True],[y[i] for i in change_True],color='green')
            plt.scatter([x[i] for i in change_False],[y[i] for i in change_False],color='red')

            #Curva de train rewrads
            current_rows=matrix_train_rewards[(n_batches*batch_size):((n_batches+1)*batch_size)]
            x=[]
            for i in range(batch_size):
                x+=list(np.linspace(n_batches*batch_size*n_steps+i*n_steps,n_batches*batch_size*n_steps+i*n_steps+n_steps,len(current_rows[i]),dtype=int))
            y=sum(current_rows,[])
            plt.plot(x,y,color='black')

            if n_batches==0:
                ax.set_title('Train rewards per episode (in black) and\n test reward per policy (dots, green when max changes)')
            elif n_batches==df_test.shape[0]//batch_size-1:
                ax.set_xlabel('Train steps')            

            n_batches+=1

    plt.savefig('experiments_getstarting/results/KnowingSingleProcess/'+str(env_name)+'/comparison_test_train_reward_on_the_fly'+str(seed)+'.pdf')
    plt.show()
    plt.close()

#==================================================================================================
# GRAFICA 4: entendiendo relacion entre train y test reward usando vectores normalizados de cambios en el maximo
#==================================================================================================
def comparison_test_train_rewards(env_name,seed,max_train_timesteps):

    # Lectura de datos
    df_train=pd.read_parquet('experiments_getstarting/results/EnvironmentProcesses/'+str(env_name)+'/df_train_'+str(env_name)+'_seed'+str(seed)+'.parquet')
    n_steps=int(df_train['n_train_timesteps'].min())

    df_test=pd.read_parquet('experiments_getstarting/results/EnvironmentProcesses/'+str(env_name)+'/df_test_'+str(env_name)+'_seed'+str(seed)+'.parquet')
    df_test=df_test[df_test['n_policy']<=max_train_timesteps//n_steps]

    trajec_train_rewards=pickle.load(open('experiments_getstarting/results/KnowingSingleProcess/'+str(env_name)+'/extracted_data/trajec_train_rewards'+str(seed)+'.pkl', 'rb'))
    window_train_rewaerds = pickle.load(open('experiments_getstarting/results/KnowingSingleProcess/'+str(env_name)+'/extracted_data/window_train_rewards'+str(seed)+'.pkl', 'rb'))
    metric_changes_norm = pickle.load(open('experiments_getstarting/results/KnowingSingleProcess/'+str(env_name)+'/extracted_data/metric_changes_norm'+str(seed)+'.pkl', 'rb'))


    # Para guardar los resultados
    output_path='experiments_getstarting/results/KnowingSingleProcess/'+str(env_name)+'/comparison_test_train_rewards'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
          
    #----------------------------------------------------------------------------------------------
    # GRAFICA 1: comparacion de train rewards "on the fly" con los test rewards
    #----------------------------------------------------------------------------------------------
    # Test reward por politica con maximo n_test_ep
    test_rewards=list(df_test['mean_reward'])
    test_reward_changes=metric_changes_norm[0]

    # Representar curva por batch
    fig=plt.figure(figsize=[10,10])
    plt.subplots_adjust(left=0.08,bottom=0.07,right=0.96,top=0.9,wspace=0.16,hspace=0.55)

    batch_size=25
    n_batches=0
    for i in range(len(test_rewards)):

        if (i+1)%batch_size==0:
            pos=n_batches+1-(i//250)*(250//25)
            ax=plt.subplot(250//batch_size,1,pos)

            # Curva de test rewards
            x=range(n_batches*batch_size*n_steps+1,(n_batches+1)*batch_size*n_steps,n_steps)
            y=test_rewards[(n_batches*batch_size):((n_batches+1)*batch_size)]
            max_change=test_reward_changes[(n_batches*batch_size):((n_batches+1)*batch_size)]
            
            change_True=[i for i in range(len(max_change)) if max_change[i]!=0]
            change_False=[i for i in range(len(max_change)) if max_change[i]==0]
    
            plt.scatter([x[i] for i in change_True],[y[i] for i in change_True],color='green')
            plt.scatter([x[i] for i in change_False],[y[i] for i in change_False],color='red')

            #Curva de train rewards
            current_rows=trajec_train_rewards[(n_batches*batch_size):((n_batches+1)*batch_size)]
            x=[]
            for j in range(batch_size):
                x+=list(np.linspace(n_batches*batch_size*n_steps+j*n_steps,n_batches*batch_size*n_steps+j*n_steps+n_steps,len(current_rows[j]),dtype=int))
            y=sum(current_rows,[])
            plt.plot(x,y,color='black')

            if pos==1:
                ax.set_title('Train rewards per episode (in black) and\n test reward per policy (dots, green when max changes)')
            elif pos==250//batch_size:
                ax.set_xlabel('Train steps')            

            n_batches+=1

        if (i+1)%250==0:
            if not os.path.exists(output_path+'/on_the_fly'+str(seed)):
                os.makedirs(output_path+'/on_the_fly'+str(seed))
            plt.savefig(output_path+'/on_the_fly'+str(seed)+'/'+str(i//250)+'.pdf')
            plt.show()
            fig=plt.figure(figsize=[10,10])
            plt.subplots_adjust(left=0.08,bottom=0.07,right=0.96,top=0.9,wspace=0.16,hspace=0.55)
    if len(test_rewards)//250<1:
        plt.savefig(output_path+'/on_the_fly'+str(seed)+'.pdf')
        ax.set_xlabel('Train steps') 
        plt.show()


    plt.close()

    #----------------------------------------------------------------------------------------------
    # GRAFICA 2: Comparar eficacia de cada metrica a la hora de detectar cambios en el maximo test reward
    #----------------------------------------------------------------------------------------------

    # Calcular vectores normalizados de cambios en el maximo reward usando las diferentes metricas de validacion
    plot_matrix=metric_changes_norm

    # Comparacion numerica entre bectores normalizados de cambios 
    overall_diff=[]
    test_reward_changes=metric_changes_norm[0]
    for i in range(0,plot_matrix.shape[0]):
        overall_diff.append(np.linalg.norm(test_reward_changes-metric_changes_norm[i]))

    diff_per_batches=[]

    # Dibujar graficas por batches de 250 politicas
    fig=plt.figure(figsize=[10,4])
    plt.subplots_adjust(left=0.13,bottom=0.152,right=0.96,top=0.9,wspace=0.16,hspace=0.2)

    for n_figure in range(plot_matrix.shape[1]//250+1):
               
        batch_plot_matrix=plot_matrix[:,(250*n_figure+1):(250*n_figure+1+250)]

        # Grafico
        ax=plt.subplot(111)
        cmap_grises = sns.color_palette("Greys", as_cmap=True)
        sns.heatmap(batch_plot_matrix, cmap=cmap_grises, cbar=True, ax=ax, linewidths=0.5, vmin=0, vmax=1,norm=PowerNorm(gamma=0.3))

        ax.set_xticks(list(range(1,250+1,20)))
        ax.set_yticks(np.arange(plot_matrix.shape[0]) + 0.5)
        ax.set_yticklabels(['Test reward', 'Trajec mean', 'Trajec max','Window 100','Window 50','Window 20','Window 10','Window 5','Window 1'],rotation=0) 
        ax.set_xticklabels(list(range(250*n_figure+1,250*n_figure+1+250,20)),rotation=0)
        colorbar=ax.collections[0].colorbar
        colorbar.set_label('Normalized change in max metric value ')
        ax.set_xlabel('n_policy')
        ax.set_ylabel('Metric to detect change')
        plt.title("Maximum change detection in metric")

        if len(test_rewards)//250>1:
            if not os.path.exists(output_path+'/overall_max_changes'+str(seed)):
                os.makedirs(output_path+'/overall_max_changes'+str(seed))
            plt.savefig(output_path+'/overall_max_changes'+str(seed)+'/'+str(n_figure)+'.pdf')
            plt.show()
            fig=plt.figure(figsize=[10,4])
            plt.subplots_adjust(left=0.13,bottom=0.152,right=0.96,top=0.9,wspace=0.16,hspace=0.2)
        else:
            plt.savefig(output_path+'/overall_max_changes'+str(seed)+'.pdf')
            plt.show()

    plt.close()

    #----------------------------------------------------------------------------------------------
    # GRAFICA 3: misma grafica anterior pero dividida en batches de politicas, para ver si 
    # el vector de cambios normalizado mas parecido al primero se da con diferenetes metricas en diferentes
    # etapas del entrenamiento. Ademas se añaden por batches los trozos correspondientes a 
    # la learning-curve por defecto y la curva de train reward por episodio train recolectado.
    #----------------------------------------------------------------------------------------------
    fig=plt.figure(figsize=[15,5])
    plt.subplots_adjust(left=0.1,bottom=0.152,right=0.96,top=0.9,wspace=0.16,hspace=0.2)

    # Normalizaciones para los trocos de curvas de las dos primeras filas
    test_rewards=[(i-min(test_rewards))/(max(test_rewards)-min(test_rewards))for i in test_rewards]
    max_value=-np.Inf
    min_value=np.Inf
    for i in trajec_train_rewards:
        if max(i)>max_value:
            max_value=max(i)
        if min(i)<min_value:
            min_value=min(i)

    # Dibujar graficas por cada 250 politicas
    batch_size=25
    n_batches=0
    for i in range(len(test_rewards)):
        if (i+1)%batch_size==0:

            # Fila 3
            batch_plot_matrix=plot_matrix[:,(25*(n_batches)+1):(25*(n_batches)+1+25)]

            # Numerico
            batch_diff=[]
            batch_test_reward_changes=batch_plot_matrix[0]
            for j in range(0,plot_matrix.shape[0]):
                batch_diff.append(np.linalg.norm(batch_test_reward_changes-batch_plot_matrix[j]))
            diff_per_batches.append(batch_diff)


            # Grafico
            ax=plt.subplot(3,250//batch_size,2*(250//batch_size)+n_batches+1-(i//250)*(250//25))
            cmap_grises = sns.color_palette("Greys", as_cmap=True)
            sns.heatmap(batch_plot_matrix, cmap=cmap_grises, cbar=False, ax=ax, linewidths=0.5, vmin=0, vmax=1,norm=PowerNorm(gamma=0.3))

            if n_batches+1-(i//250)*(250/25)==1:
                ax.set_yticks(np.arange(plot_matrix.shape[0]) + 0.5)
                ax.set_yticklabels(['Test reward', 'Trajec mean', 'Trajec max','Window 100','Window 50','Window 20','Window 10','Window 5','Window 1'],rotation=0) 
                ax.set_ylabel('Validation metric\n(Darker = Greater Change\nin Max Reward)')

            else:
                ax.set_yticklabels([])
            
            ax.set_xlabel('Batch'+str(n_batches+1)+'\nof '+str(batch_size)+'policies')
            ax.set_xticklabels([])

            # Fila 2
            ax=plt.subplot(3,250//batch_size,(250//batch_size)+n_batches+1-(i//250)*(250//25))

            x=range(n_batches*batch_size+1,(n_batches+1)*batch_size+1)
            y=test_rewards[(n_batches*batch_size):((n_batches+1)*batch_size)]
            y=[(i-min(test_rewards))/(max(test_rewards)-min(test_rewards))for i in y]
            plt.plot(x,y,color='grey')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_ylim(-0.1,1.1)

            if n_batches+1-(i//250)*(250//25)==1:
                ax.set_ylabel('Test reward\nlast visited policy\n(normalized)')

            # Fila 1
            ax=plt.subplot(3,250//batch_size,n_batches+1-(i//250)*(250//25))

            current_rows=trajec_train_rewards[(n_batches*batch_size):((n_batches+1)*batch_size)]
            x=[]
            for j in range(batch_size):
                x+=list(np.linspace(n_batches*batch_size+j,n_batches*batch_size+j+1,len(current_rows[j]),dtype=int))
            y=sum(current_rows,[])
            y=[(i-min_value)/(max_value-min_value) for i in y]
            plt.plot(x,y,color='grey')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_ylim(-0.1,1.1)

            if n_batches+1-(i//250)*(250//25)==1:
                ax.set_ylabel('Train rewards\nper episode\n(normalized)')

            n_batches+=1

        if (i+1)%250==0: 
            if not os.path.exists(output_path+'/per_batches'+str(seed)):
                os.makedirs(output_path+'/per_batches'+str(seed))
            plt.savefig(output_path+'/per_batches'+str(seed)+'/'+str(i//250)+'.pdf')
            plt.show()
            fig=plt.figure(figsize=[15,5])
            plt.subplots_adjust(left=0.1,bottom=0.152,right=0.96,top=0.9,wspace=0.16,hspace=0.2)
    if len(test_rewards)//250<1:
        plt.savefig(output_path+'/per_batches'+str(seed)+'.pdf')
        plt.show()

    plt.close()

    # Guardar tabla con datos de diferencia entre vectores normalizados
    validation_metrics=['Test reward', 'Trajec mean', 'Trajec max','Window 100','Window 50','Window 20','Window 10','Window 5','Window 1']
    info_table=[]
    for i in range(len(validation_metrics)):
        info_table.append([validation_metrics[i],overall_diff[i],np.mean(np.array(diff_per_batches)[:,i])])
    info_table=pd.DataFrame(info_table,columns=['metric','overall_diff','mean_diff_batch'])
    info_table.to_csv(output_path+'/info_table.csv')


#==================================================================================================
# Programa principal
#==================================================================================================

# Inverted Double pendulum
#--------------------------------------------------------------------------------------------------
env_name='InvertedDoublePendulum'

# Calculos iniciales (para no repetirlos despues)
extract_validation_data_from_dfs(env_name,3,500000)
extract_validation_data_from_dfs(env_name,11,500000)

# Validacion con test reward considerando diferentes acc y freq
learning_curves_test_reward(list(range(10000, 500001, 5000)),[100,50,20,10,5,1],env_name,3)
learning_curves_test_reward(list(range(10000, 500001, 5000)),[100,50,20,10,5,1],env_name,11)
learning_curves_test_reward(list(range(10000, 500001, 5000)),[100,50,20,10,5,1],env_name,3,[5])
learning_curves_test_reward(list(range(10000, 500001, 5000)),[100,50,20,10,5,1],env_name,3,[10])
learning_curves_test_reward(list(range(10000, 500001, 5000)),[100,50,20,10,5,1],env_name,3,[50])
learning_curves_test_reward(list(range(10000, 500001, 5000)),[1],env_name,3,[1,5,10,50])

# Validacion con train reward usando diferentes metricas
learning_curves_train_reward(list(range(10000, 500001, 5000)),env_name,3)

# Encontrando relacion entre train reward y test reward
comparison_test_train_rewards(env_name,3,500000)
# comparison_test_train_rewards_rankings([100,50,25,12,6,3,1],env_name,3,500000)

# Usar train rewards para definir la frecuencia de la validacion con test reward
metrics_train_changes=pickle.load(open('experiments_getstarting/results/KnowingSingleProcess/InvertedDoublePendulum/extracted_data/metric_changes_norm3.pkl', 'rb'))
learning_curves_test_reward(list(range(10000, 500001, 5000)),[1],env_name,3,[1,5,10,20,['Defined by max change in "Trajec max"',metrics_train_changes[2]]])
learning_curves_test_reward(list(range(10000, 500001, 5000)),[5],env_name,3,[1,5,10,20,['Defined by max change in "Trajec max"',metrics_train_changes[2]]])


# Ant
#--------------------------------------------------------------------------------------------------
env_name='Ant'

# Calculos iniciales (para no repetirlos despues)
extract_validation_data_from_dfs(env_name,1,3200000)

# Validacion con test reward considerando diferentes acc y freq
learning_curves_test_reward(list(range(10000, 3200001, 32100)),[100,50,20,10,5,1],env_name,1)
learning_curves_test_reward(list(range(10000, 3200001, 32100)),[100,50,20,10,5,1],env_name,1,[5])
learning_curves_test_reward(list(range(10000, 3200001, 32100)),[100,50,20,10,5,1],env_name,1,[10])
learning_curves_test_reward(list(range(10000, 3200001, 32100)),[100,50,20,10,5,1],env_name,1,[20])
learning_curves_test_reward(list(range(10000, 3200001, 32100)),[1],env_name,1,[1,5,10,20])

# Validacion con train reward usando diferentes metricas
learning_curves_train_reward(list(range(10000, 3200001, 32100)),env_name,1)

# Encontrando relacion entre train reward y test reward
comparison_test_train_rewards(env_name,1,3200000)

# Usar train rewards para definir la frecuencia de la validacion con test reward
metrics_train_changes=pickle.load(open('experiments_getstarting/results/KnowingSingleProcess/Ant/extracted_data/metric_changes_norm1.pkl', 'rb'))
learning_curves_test_reward(list(range(10000, 3200001, 32100)),[1],env_name,1,[1,5,10,20,['Defined by max change in "Trajec mean"',metrics_train_changes[1]]])
learning_curves_test_reward(list(range(10000, 3200001, 32100)),[5],env_name,1,[1,5,10,20,['Defined by max change in "Trajec mean"',metrics_train_changes[1]]])


