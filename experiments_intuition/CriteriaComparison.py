'''TODO: este encabezado fue actualizado tras el MAEB (ahora hay mas)

En este scrip se aborda la comparacion grafica de los criterios existentes.

Se consideran 30 procesos de aprendizaje: 

- seed=1,...,10
algo= PPO, env=Ant, learning time=10000000 steps, 16 CPU para interaccion train y validacion en 1000 episodios (y device='auto')

- seed=1,...,10
algo= PPO, env=Humanoid, learning time=10000000 steps, 16 CPU para interaccion train y validacion en 1000 episodios (y device='auto')

- seed=1,...,10
algo= PPO, env=HumanoidStandup, learning time=10000000 steps, 16 CPU para interaccion train y validacion en 1000 episodios (y device='auto')


De los 1000 datos de episodic reward almacenados:
- 500 para ground truth
- 500 para simular diferentes tamaños de episodios de validacion

Las graficas para procesos de aprendizaje indepedientes (definidos por un algoritmo, environment, semilla y tiempo maximo) representan:
- Proceso de aprendizaje con learning-curves, nivel de degradacion y degradacion por actualizacion.
- Ajuste de la configuracion optima para los criterios con parametros.
- Comparacion de criterios con evolucion de rank y magnitud.
- Coste y precision de estimaciones para la seleccion.

Las graficas para comparacion de criterios segun degradacion (independiente al proceso) representan:
- Comparacion de los 3 criterios (last, best train, best test) en su mejor version por nivel de degradacion.
- Comparacion de criterios train y test para diferentes configuraciones (las registradas como optimas para procesos individuales) por nivel de degradacion. (analisis de sensibilidad)
- Comparacion de criterios last, train y test por tiempos de ejecucion (cantidad de iteraciones de entrenamiento).

TODO: comparar la eficacia de los criterios dependiendo del nivel de paralelizacion usado para aprender y validar (mas adelante, ahora 16 CPU con 1 GPU).
- Cuantos menos CPU-> mayor degradacion (?)
- Cuantos menos CPU-> mayor diferencia en validation time vs. interaction time (?)
- Cuantos menos CPU-> peores estimaciones train vs. estimaciones validacion (?)

NOTE: aunque SB3 no este pensado para aprovechar los GPU, he observado que asignandole GPUs va mas rapido.
'''

import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import json
import bz2
import base64
import os
import numpy as np
from tqdm import tqdm
import matplotlib.colors as mcolors
import math
import matplotlib.patches as patches
from itertools import chain
import csv
import re
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
import seaborn as sns
from matplotlib.ticker import FuncFormatter, FormatStrFormatter, MultipleLocator



parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))



class Estimator:
    '''
    Las funciones de esta clase permiten hacer estimaciones asociadas a cada iteracion de un proceso a partir de los datos train o test
    almacenados durante la ejecucion de un proceso definido por (algoritmo RL-environmnet-seed-tiempo maximo). Esas estimaciones pueden ser:
    1) Estimaciones de expected episodic reward
    2) Estimaciones de degradacion
    '''
    # TODO: esta funcion igual encaja mas en Converter
    def time_discretizer(algo,env,seed,resources,iter_freq,iter_max,min_time):
        '''
        Devuelve una discretizacion del vector de tiempos de entrenamiento en segundos (eje OX). 
        Util cuando queremos medir el tiempo en segundos y no en steps.
        '''

        # Leer bases de datos train del proceso de interes
        current_path=parent_dir+'/_bender/project_SB3/data/'+algo+'_'+env+'_seed'+str(seed)+'_'+resources
        df_train=pd.read_csv(current_path+'/df_traj.csv')

        # Tiempo medio por iteracion
        time_between_iter=[df_train['time_seconds'][i+1]-df_train['time_seconds'][i] for i in range(df_train.shape[0]-1)]
        iter_mean_time=int(np.mean(time_between_iter))

        # Vector de tiempos discretizado con la frecuencia indicada y el valor maximo como limite
        return list(np.arange(min_time,df_train.loc[iter_max-1,'time_seconds'],iter_mean_time*iter_freq))

    #----------------------------------------------------------------------------------------------
    # Para estimaciones de expected episodic reward
    #----------------------------------------------------------------------------------------------
    def estimate_from_traj(train_rewards_workers,train_ep_ends_workers,n_timesteps_per_iter,n_traj_ep):

        '''
        Asi se calcula, por las librerias, la estimacion del expected episodic reward usando las trayectorias de train,
        cuando hacemos ejecuciones en paralelo (multiples worker).

        Esta funcion calcula la estimacion para todas las politicas de la secuencia.

        `train_rewards`: matriz de rewards de trayectorias concatenados en orden de creacion (una fila por worker)
        `train_ep_ends`: matriz de dones de trayectorias concatenados en orden de creacion (una fila por worker)
        `n_timesteps_per_iter`: lista con los steps consumidos por iteracion en paralelo (si hay mas de un worker, se tiene en cuenta el rollout)
        `n_traj_ep`: numero de episodios previos para calcular la media
        '''

        def ER_seq_single_worker(train_rewards,train_ep_ends,time_steps):
            '''
            Devuelve lista de los episodic reward (ER) acumulados en las trayectorias, y tiempos (en steps) en los que los episodios se completan.
            '''
            current_train_rewards=train_rewards[:time_steps]
            current_train_ep_ends=train_ep_ends[:time_steps]

            ep_rw=[]
            ep_last=[]
            last_i=0
            current_i=0

            for done in current_train_ep_ends:
                current_i+=1
                if done:
                    ep_rw.append(sum(current_train_rewards[last_i:current_i]))
                    ep_last.append(current_i)
                    last_i=current_i

            return ep_rw, ep_last
        
        ep_rw_policy=[]
        for time_steps in tqdm(n_timesteps_per_iter):
            ep_rw_workers=[]
            ep_last_workers=[]

            # Guardar datos de rewards por episodio y cuando se han almacenado para cada worker
            for i in range(len(train_rewards_workers)):
                ep_rw, ep_last=ER_seq_single_worker(train_rewards_workers[i],train_ep_ends_workers[i],time_steps)
                ep_rw_workers+=ep_rw
                ep_last_workers+=ep_last

            # Quedarnos con los n_traj_ep ultimos teniendo en cuenta el tiempo de almacenamiento
            ep_rw_sorted = [x for _, x in sorted(zip(ep_last_workers, ep_rw_workers))] # ordenar ep_rw segun ep_last

            if len(ep_rw_sorted)<n_traj_ep:
                ep_rw_policy.append(np.mean(ep_rw_sorted))
            else:
                ep_rw_policy.append(np.mean(ep_rw_sorted[-n_traj_ep:]))

        return ep_rw_policy

    #----------------------------------------------------------------------------------------------
    # Para estimaciones de degradacion
    #----------------------------------------------------------------------------------------------
    def estimate_any_degradation(A,B,degradation_metric,also_dominance=False,additionals=None):
        '''
        `degradation_metric`: 'greater_prob', 'paired_diff_probpos_meanpos', 'paired_diff_median', 'paired_diff_probpos', 'relative_reward_diff'
        
        'greater_prob'
        Transformacion de la estimacion de la probabilidad de que las variables aleatorias de las que provienen las muestras A y B cumplan: X_A < X_B 
        https://www.tandfonline.com/doi/full/10.1080/10618600.2022.2084405

        Transformacion de dominancia C_p para definir nuestra degradacion por actualizacion (valor en [0,1], 0 no hay degradacion y 
        cuanto mas cerca de 1 mas degradacion). Esta funcion normaliza C_p en [0,1] cuando C_p toma valores en (0.5,1] (i.e. B=X_prev domina
        a A=X_current) y 0 en los demas casos (A=X_current domina a B=X_prev o son iguales).

        'paired_diff_probpos'
        Estimacion de que la probabilidad de la variable aleatoria definida como la diferencia pareada de los episodic reward de dos politicas en
        la secuencia sea positiva.
        '''
        if degradation_metric=='greater_prob':  
            # Estimacion de Cp
            sign_matrix = np.sign(B[:, None] - A)  # Matriz de comparacion
            cp_estimation=(np.sum(sign_matrix) / (2 * len(A)**2)) + 0.5

            # Modificacion para que se defina en [0,1]
            indicator=[1 if cp_estimation>0.5 else 0][0]
            degradation=2*(cp_estimation-0.5)*indicator

            if also_dominance:
                return degradation, cp_estimation
            else:
                return degradation
        
        if degradation_metric=='paired_diff_probpos_meanpos':
            # Normalizar los valores de las dos variables para que la metrica salga en [0,1]
            max_value=max([max(A),max(B)])
            min_value=min([min(A),min(B)])
            norm_A=Converter.normalize_list(A,min_value,max_value)
            norm_B=Converter.normalize_list(B,min_value,max_value)

            # Variable de diferencias pareadas
            paired_diffs=norm_B-norm_A

            # Media de diferencias positivas ponderada por la proporcion de positivos
            degradation=np.mean(paired_diffs>0)*np.mean([diff for diff in paired_diffs if diff>0])

            return degradation

        if degradation_metric=='paired_diff_median':
            # Normalizar los valores de las dos variables para que la metrica salga en [0,1]
            max_value=max([max(A),max(B)])
            min_value=min([min(A),min(B)])
            norm_A=Converter.normalize_list(A,min_value,max_value)
            norm_B=Converter.normalize_list(B,min_value,max_value)

            # Variable de diferencias pareadas
            paired_diffs=norm_B-norm_A

            # Media de diferencias positivas ponderada por la proporcion de positivos
            degradation=np.median(paired_diffs)# este valor esta definido en [-1,1]
            degradation=[degradation if degradation>0 else 0][0]

            return degradation
        
        if degradation_metric=='paired_diff_probpos':

            # Variable de diferencias pareadas
            paired_diffs=B-A

            # Proporcion de positivos en la diferencia ponderada
            degradation=np.mean(paired_diffs>0)

            return degradation
        
        if degradation_metric=='relative_reward_diff':
            # NOTE: esta metrica es la unica entre todas que no usa la distribucion (muestra) de los rewards, unicamente usa la media.
            #Por ello, para qe la degradacion local sea un valor en [0,1] y comparable con otras degradaciones locales de otro par
            #de politicas asociado otras tareas/procesos, debemos normalizarla/estandarizarla. Para ello necesitamos conocer el reward maximo
            #y el reward minimo alcanzable en esa tarea (cosa que desconocemos). Como nosotros solo usaremos esta medida para comparar
            #diferentes pares de politicas en el mismo proceso, aproximaremos esos valores con el maximo y minimo observados en el proceso.
            #Por tanto, la  metrica resultante es equivalente a la magnitud.

            indicator=[1 if np.mean(B)>np.mean(A) else 0][0]
            relative_diff=abs(np.mean(A)-np.mean(B))/(additionals[1]-additionals[0])
            degradation=relative_diff*indicator

            return degradation

        
    def estimate_update_degradations(algo,env,seed,resources, iter_max,degradation_metric='greater_prob',additionals=None):
        '''
        Calcula la evolucion de la degradacion local entre politicas consecutivas, i.e. update degradation, usando la metrica local indicada.
        '''
        current_path=parent_dir+'/_bender/project_SB3/data/'+str(algo)+'_'+str(env)+'_seed'+str(seed)+'_'+resources

        # Leer base de datos test
        df_test=pd.read_csv(current_path+'/df_val.csv')
        df_test['ep_rewards']=[Converter.compress_decompress_list(i,compress=False) for i in df_test['ep_rewards']][:iter_max]

        # Calcular vector de degradaciones por actualizacion
        update_degradations=[0] # En la inicializacion de la secuencia de politicas no hay degradacion
        update_dominances=[0]
        for row in range(1,iter_max):
            X_current=df_test.loc[row,'ep_rewards'][:500] 
            X_prev=df_test.loc[row-1,'ep_rewards'][:500]

            if degradation_metric=='greater_prob':
                update_degradation,dominance=Estimator.estimate_any_degradation(np.array(X_current),np.array(X_prev),degradation_metric,also_dominance=True)
                update_degradations.append(update_degradation)
                update_dominances.append(dominance)
            if degradation_metric in ['paired_diff_probpos_meanpos','paired_diff_median','paired_diff_probpos']:
                update_degradations.append(Estimator.estimate_any_degradation(np.array(X_current),np.array(X_prev),degradation_metric))
            if degradation_metric=='relative_reward_diff':
                update_degradations.append(Estimator.estimate_any_degradation(np.array(X_current),np.array(X_prev),degradation_metric,
                                                                              additionals=additionals))

        if degradation_metric=='greater_prob':
            return update_degradations,update_dominances
        if degradation_metric in ['paired_diff_probpos_meanpos','paired_diff_median','paired_diff_probpos','relative_reward_diff']:
            return update_degradations

    #----------------------------------------------------------------------------------------------
    # Para generar dos bases de datos adicionales (una para train y otra para test) con las
    # diferentes posibles estimaciones a partir de los datos train o test
    #----------------------------------------------------------------------------------------------
    def read_create_estimates_csv(path_csv,seq_size,start_iter=0):
        '''
        Crea (si no existe) o lee (si existe) una base de datos para almacenar estimaciones de expected episodic reward por politica de la secuencia.
        '''
        if not os.path.exists(path_csv):
            df_estimates = pd.DataFrame({"n_policy": list(range(start_iter,seq_size))})
            os.makedirs(os.path.dirname(path_csv), exist_ok=True)
            df_estimates.to_csv(path_csv, index=False)
        else:
            df_estimates=pd.read_csv(path_csv)

        return df_estimates
    
    def compute_estimates(algo,env,seed,resources,n_ep,train_test_estimate):
        '''
        Antes de generar las graficas, se calculan las estimaciones de expected episodic reward (EER) con train y test.
        Las estimaciones se almacenan en bases de datos adicionales. Asi se pueden calcular tantas estimaciones como
        curvas se quieran dibujar antes de dibujar las curvas (e.g. diferentes numeros de episodios para la estimacion train).
        Esto permitira no tener que hacer los mismos calculos multiples veces.

        Cuando las estimaciones se hacen con datos de validacion, no solo se calculan las estimaciones de EER, tambien el
        tiempo adicional necesario en segundos.
        '''
        
        # Leer bases de datos del proceso de interes
        current_path=parent_dir+'/_bender/project_SB3/data/'+algo+'_'+env+'_seed'+str(seed)+'_'+resources
        df_train=pd.read_csv(current_path+'/df_traj.csv')
        df_test=pd.read_csv(current_path+'/df_val.csv')
        
        # Calcular estimaciones que usaremos como ground truth con los primeros 500 episodios de validacion
        df_test['ep_rewards']=[ Converter.compress_decompress_list(i,compress=False) for i in df_test['ep_rewards']]
        df_val_estimates=Estimator.read_create_estimates_csv(current_path+'/df_val_estimates.csv',df_train.shape[0])
        #print([ np.mean(i[:500]) for i in df_test['ep_rewards'] ]==df_val_estimates['truth'].tolist()) NOTE: tras pasar un tiempo y volver a repetir la operacion de descomprension y media, los resultados salen diferentes
        
        if 'truth' not in df_val_estimates.columns.tolist():
            df_val_estimates['truth']=[ np.mean(i[:500]) for i in df_test['ep_rewards'] ]
            df_val_estimates.to_csv(current_path+'/df_val_estimates.csv', index=False)
        if 'truth_norm' not in df_val_estimates.columns.tolist():
            df_val_estimates['truth_norm']=[ (i-min(df_val_estimates['truth']))/(max(df_val_estimates['truth'])-min(df_val_estimates['truth'])) for i in df_val_estimates['truth'] ]
            df_val_estimates.to_csv(current_path+'/df_val_estimates.csv', index=False)
        
        # Tambien estimaciones de degradacion para añadir a las learning-curve mas informativas (empezar a considerar la degradacion despues del 10% del tiempo)
        degradation_metrics=['greater_prob','paired_diff_probpos_meanpos','paired_diff_median','paired_diff_probpos','relative_reward_diff']
        for degradation_metric in degradation_metrics:
            if 'update_deg_'+degradation_metric not in df_val_estimates.columns.tolist():
                if degradation_metric=='greater_prob':
                    update_degradations,update_dominances=Estimator.estimate_update_degradations(algo,env,seed,resources,df_test.shape[0],degradation_metric)
                    df_val_estimates['update_dominances']=[0 for _ in range(int(df_test.shape[0]*.1))]+update_dominances[int(df_test.shape[0]*.1):]
                if degradation_metric in ['paired_diff_probpos_meanpos','paired_diff_median','paired_diff_probpos']:
                    update_degradations=Estimator.estimate_update_degradations(algo,env,seed,resources,df_test.shape[0],degradation_metric)
                if degradation_metric=='relative_reward_diff':
                    update_degradations=Estimator.estimate_update_degradations(algo,env,seed,resources,df_test.shape[0],degradation_metric,
                                                                               additionals=[df_val_estimates['truth'].min(),df_val_estimates['truth'].max()])

                df_val_estimates['update_deg_'+degradation_metric]=[0 for _ in range(int(df_test.shape[0]*.1))]+update_degradations[int(df_test.shape[0]*.1):]
                df_val_estimates.to_csv(current_path+'/df_val_estimates.csv', index=False)

        # Calcular estimaciones a partir de datos de train
        df_traj_estimates=Estimator.read_create_estimates_csv(current_path+'/df_traj_estimates.csv',df_train.shape[0])
        if train_test_estimate=='train':
            # Añadir columnas de interes en df_traj_estimates (siempre que ya no esten calculadas previamente)
            df_train['traj_rewards']=[ Converter.compress_decompress_list(i,compress=False) for i in df_train['traj_rewards']]
            df_train['traj_ep_end']=[ Converter.compress_decompress_list(i,compress=False) for i in df_train['traj_ep_end']]
            rollout=np.array(df_train['traj_rewards'][0]).shape[1]
            
            if str(n_ep)+'_traj_ep' not in df_traj_estimates.columns.tolist():
                df_traj_estimates[str(n_ep)+'_traj_ep']=Estimator.estimate_from_traj(
                    Converter.concat_traj_seq(df_train['traj_rewards']),
                    Converter.concat_traj_seq(df_train['traj_ep_end']),
                    list(range(rollout,rollout*(df_train.shape[0]+1),rollout)),n_ep)
                df_traj_estimates.to_csv(current_path+'/df_traj_estimates.csv', index=False)

        # Calcular estimaciones a partir de datos de validacion
        if train_test_estimate=='test':
            # Añadir columnas de interes en df_val_estimates (siempre que ya no esten calculadas previamente)
            if str(n_ep)+'_val_ep' not in df_val_estimates.columns.tolist():
                estimates=[np.mean(i[500:(500+n_ep)]) for i in df_test['ep_rewards']]
                val_times=[]
                for i in range(df_test.shape[0]):
                    df_test_elapsed_val_time=Converter.compress_decompress_list(df_test['elapsed_val_time'][i],compress=False)
                    df_test_n_val_ep=Converter.compress_decompress_list(df_test['n_val_ep'][i],compress=False)
                    val_time_until_truth=df_test_elapsed_val_time[df_test_n_val_ep.index(500)] 
                    val_time=df_test_elapsed_val_time[df_test_n_val_ep.index(500+n_ep)] 
                    val_times.append(val_time-val_time_until_truth)
                          
                df_val_estimates[str(n_ep)+'_val_ep']=[Converter.compress_decompress_list(i) for i in zip(estimates,val_times)]
                df_val_estimates.to_csv(current_path+'/df_val_estimates.csv', index=False)
        
 
class Converter:
    '''
    Las funciones de esta clase sirven para hacer conversiones, i.e. convertir el formato original de un input en otro formato output.
    '''

    def compress_decompress_list(my_list,compress=True):
        '''
        Sirve para comprimir o descomprimir listas almacenadas en las columnas de las bases de datos.
        '''
        if compress:
            # Convertir la lista a una cadena JSON compacta
            json_str = json.dumps(my_list)

            # Comprimir la cadena
            compressed_data = bz2.compress(json_str.encode('utf-8'))

            # Convertir a base64 para guardar como texto en la base de datos
            compressed_str = base64.b64encode(compressed_data).decode('utf-8')

            return compressed_str
        else:
            # Leer la cadena comprimida de la base de datos
            compressed_data = base64.b64decode(my_list.encode('utf-8'))

            # Descomprimir la cadena
            json_str = bz2.decompress(compressed_data).decode('utf-8')

            # Convertir la cadena JSON de vuelta a lista
            my_list = json.loads(json_str)

            return my_list
        
    def concat_traj_seq(traj_seq):
        '''
        Concatenar trayectorias, proveniente de multiples workers y obtenidas con las diferentes politicas de la secuencia
        '''
        return [np.concatenate(row, axis=0) for row in zip(*np.array(traj_seq))]

    def normalize_list(lst,my_min=None,my_max=None):
        '''
        Normaliza la lista proporcionada para que el valor maximo en ella sea 1 y el valor minimo 0.
        '''
        lst = np.array(lst)

        if my_min==None and my_max==None:
            return (lst - np.min(lst)) / (np.max(lst) - np.min(lst))
        else:
            return (lst - my_min) / (my_max - my_min)
        
    def generate_colormap(value, cmap_name, vmin, vmax):
        """Asigna un color segun el valor y el mapa de colores indicado"""
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap(cmap_name)
        return cmap(norm(value))

    def generate_colorbar(fig,position_size,cmap,data_range,cbar_title,orientation='vertical'):
        '''Crear barra de color para la escala de grises invertida
        `position_size`: [left, bottom, width, height] posicion y tamaño de la barra de colores
        `data_range`: [vmin,vmax] valor minimo y maximo de los datos que se quieren dibujar
        `orientation`: 'vertical' o 'horizontal', orientacion de la barra de colores
        '''
        cbar_ax = fig.add_axes(position_size)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=data_range[0], vmax=data_range[1]))  # Usar 'Blues' para las filas azules
        sm.set_array([])

        if orientation=='vertical':
            cbar = plt.colorbar(sm, cax=cbar_ax)
            cbar.set_label(cbar_title,fontsize=12)
            cbar.ax.yaxis.set_tick_params(rotation=90)
        if orientation=='horizontal':
            cbar = plt.colorbar(sm, cax=cbar_ax,orientation='horizontal')
            cbar.set_label(cbar_title,fontsize=12, labelpad=5)
            cbar.ax.xaxis.set_tick_params(rotation=0)

    def generate_df(path_csv,column_names):

        if not os.path.exists(path_csv):
            df_estimates = pd.DataFrame(columns=column_names)
            os.makedirs(os.path.dirname(path_csv), exist_ok=True)
            df_estimates.to_csv(path_csv, index=False)
        else:
            df_estimates=pd.read_csv(path_csv)
        
        return df_estimates

    def from_csv_to_png(csv_path,csv_name):
        df = pd.read_csv(csv_path+'/'+csv_name+'.csv')
        df=df.sort_values(by='process_id', ascending=True)

        # Create a figure and axis to plot the table
        plt.figure(figsize=(8, 3))  # Adjust the size as needed
        ax = plt.gca()

        # Hide axes
        ax.axis('tight')
        ax.axis('off')

        # Render the table in the plot and save the figure as an image
        plt.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
        plt.savefig(csv_path+'/'+csv_name+'.pdf', bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def process_id_splitter(process_id):
        parts = process_id.split('_')  # Dividir por guion bajo
        parts[-1] = int(re.search(r'\d+', parts[-1]).group())  # Convertir el ultimo elemento en numero (la semilla)
        return parts
    
    def from_list_to_ranking(value_list,small_best=True):
        indxs=np.argsort(value_list)
        ranking=[None]*len(value_list)
        for i in range(len(value_list)):
            ranking[indxs[i]]=i+1
        return ranking
    
    def from_list_to_prob_not_first(value_list,small_best=True):
        value_list=list(value_list)
        min_value=[min(value_list) if min(value_list)!=0 else 1e-10][0]
        value_list.remove(min(value_list))
        return max([min_value/value if value!=0 else 1 for value in value_list])
    
    def from_list_to_diff_not_best(value_list,small_best=True):
        value_list=list(value_list)
        min_value= min(value_list)
        value_list.remove(min_value)
        next_min_value=min(value_list)

        if next_min_value==0:
            return 0
        else:
            return (next_min_value-min_value)/next_min_value
    
    def from_lists_best_worst_to_relative_diff(best_mag,worst_mag):
        diff=[]
        for mag1,mag2 in zip(best_mag,worst_mag):

            if mag2==0:
                diff.append(0)
            else:
                diff.append((mag2-mag1)/mag2)

        return diff


class EvolutionGenerator:
    '''
    Esta clase contiene funciones que permiten generar la evolucion de diferentes metricas durante un proceso determinado por
    algortimo-environment-semilla. 
    '''
    def __init__(self,algo,env,seed,resources,perc_time_start):
        '''
        `algo`: algortimo RL usado para genera el proceso (primer nombre separado por '_' de la carpeta en donde se almacenan los datos)
        `env`: environmnet
        `seed`: semilla aleatoria
        `resources`: recursos computacionales para ejecucion en paralelo 
        `perc_time_start`: que porcentage inicial de tiempo de entrenamiento se va a ignorar para empezar a valorar la eficacia de los criterios
        '''
          
        # Leer bases de datos del proceso de interes
        current_path=parent_dir+'/_bender/project_SB3/data/'+str(algo)+'_'+str(env)+'_seed'+str(seed)+'_'+resources
        df_train=pd.read_csv(current_path+'/df_traj.csv')
        df_train_estimates=pd.read_csv(current_path+'/df_traj_estimates.csv')
        df_test=pd.read_csv(current_path+'/df_val.csv')
        df_test_estimates=pd.read_csv(current_path+'/df_val_estimates.csv')

        self.df_train=df_train
        self.df_test=df_test
        self.df_train_estimates=df_train_estimates
        self.df_test_estimates=df_test_estimates
        self.start_iter=int(df_train.shape[0]*perc_time_start)
        self.start_time=df_train.loc[self.start_iter-1,'time_seconds']

    #----------------------------------------------------------------------------------------------
    # Metricas para medir la calidad de la politica seleccionada o nivel de degradacion de la secuencia
    #----------------------------------------------------------------------------------------------
    def real_ranking_position(self,elapsed_time,n_policy):

        last_policy=self.last_policy(elapsed_time)
        EER_list=self.df_test_estimates[(self.df_test['n_policy']<=last_policy) & (self.df_test['n_policy']>=self.start_iter-1)]['truth'].tolist()

        return np.argsort(EER_list)[::-1].tolist().index(n_policy-self.start_iter+1)

    def magnitude(self,elapsed_time,n_policy,normalized):

        last_policy=self.last_policy(elapsed_time)
        if not normalized:
            EER_list=self.df_test_estimates[(self.df_test['n_policy']<=last_policy) & (self.df_test['n_policy']>=self.start_iter-1)]['truth'].tolist()
        if normalized:
            EER_list=self.df_test_estimates[(self.df_test['n_policy']<=last_policy) & (self.df_test['n_policy']>=self.start_iter-1)]['truth_norm'].tolist()

        EER_real_best=max(EER_list)
        EER_criteria_best=EER_list[n_policy-self.start_iter+1]

        return abs(EER_real_best-EER_criteria_best)
    
    def degradation_level(self,elapsed_time,global_metric,local_metric):
        '''
        `global_metric`: 'mean_update_deg', 'weighted_mean_best_later_deg', 'best_last_deg', 'relative_worsening_to_improvement'
        `local_metric`: 'greater_prob', 'paired_diff_probpos_meanpos', 'paired_diff_median', 'paired_diff_probpos', 'relative_reward_diff'
        '''

        if global_metric=='mean_update_deg':
            last_idx=self.last_policy(elapsed_time)
            degradation_level=np.mean(self.df_test_estimates['update_deg_'+local_metric][int(self.df_test.shape[0]*.1)-1:last_idx+1])

        if global_metric=='weighted_mean_best_later_deg':
            # Indice de la mejor truth politica
            truth_best_idx=self.truth_best_policy(elapsed_time)
            last_idx=self.last_policy(elapsed_time)

            # Guardar: 1) Degradaciones entre cada politica posterior a la mejor y la mejor; 2) Numero de iteraciones (normalizadas) trasncurridas entre cada politica posterior a la mejor y la mejor
            degradations_best_laters=[]
            iter_diff_best_laters=[]
            best_ep_rewards=np.array(Converter.compress_decompress_list(self.df_test.loc[truth_best_idx,'ep_rewards'],compress=False)[:500])
            for later_idx in range(truth_best_idx+1,last_idx+1):
                later_ep_rewards=np.array(Converter.compress_decompress_list(self.df_test.loc[later_idx,'ep_rewards'],compress=False)[:500])
                degradations_best_laters.append(Estimator.estimate_any_degradation(later_ep_rewards,best_ep_rewards,local_metric))
                iter_diff_best_laters.append(later_idx-truth_best_idx)
            iter_diff_best_laters=np.array(iter_diff_best_laters)/sum(iter_diff_best_laters)

            # Nivel de degradacion: media de las degradaciones ponderada por los numeros de iteraciones trasncurridos
            degradation_level=0
            for weight, deg in zip(iter_diff_best_laters,degradations_best_laters):
                degradation_level+=weight*deg
        
        if global_metric=='best_last_deg':
            # Indices de la ultima y mejor politica
            truth_best_idx=self.truth_best_policy(elapsed_time)
            last_idx=self.last_policy(elapsed_time)

            # Degradacion local entre ellas
            best_ep_rewards=np.array(Converter.compress_decompress_list(self.df_test.loc[truth_best_idx,'ep_rewards'],compress=False)[:500])
            last_ep_rewards=np.array(Converter.compress_decompress_list(self.df_test.loc[last_idx,'ep_rewards'],compress=False)[:500])
            degradation_level=Estimator.estimate_any_degradation(last_ep_rewards,best_ep_rewards,local_metric)

        if global_metric=='relative_worsening_to_improvement':
            # Indices de la ultima y mejor politica en el tiempo transcurrido
            truth_best_idx=self.truth_best_policy(elapsed_time)
            last_idx=self.last_policy(elapsed_time)

            # Identificar la peor y mejor politicas del proceso (para la normmalizacion y que la metrica local este en [0,1])
            process_worst_ep_truth=self.df_test_estimates['truth'].min()
            process_best_ep_truth=self.df_test_estimates['truth'].max()

            # Degradacion local entre: mejor-ultima y mejor-primera
            best_ep_truth=self.df_test_estimates.loc[truth_best_idx,'truth']
            last_ep_truth=self.df_test_estimates.loc[last_idx,'truth']
            init_ep_truth=self.df_test_estimates.loc[self.start_iter-1,'truth']

            best_last_deg=Estimator.estimate_any_degradation(last_ep_truth,best_ep_truth,local_metric,additionals=[process_worst_ep_truth,process_best_ep_truth])
            best_init_deg=Estimator.estimate_any_degradation(init_ep_truth,best_ep_truth,local_metric,additionals=[process_worst_ep_truth,process_best_ep_truth])
            indicator=[0 if init_ep_truth!=last_ep_truth else 1][0]

            degradation_level=best_last_deg/max(best_init_deg,best_last_deg,indicator)
        return degradation_level
    
    def effectiveness(self,elapsed_time,n_policy,normalized):

        last_policy=self.last_policy(elapsed_time)
        if not normalized:
            EER_list=self.df_test_estimates[(self.df_test['n_policy']<=last_policy) & (self.df_test['n_policy']>=self.start_iter-1)]['truth'].tolist()
        if normalized:
            EER_list=self.df_test_estimates[(self.df_test['n_policy']<=last_policy) & (self.df_test['n_policy']>=self.start_iter-1)]['truth_norm'].tolist()

        EER_real_best=max(EER_list)
        EER_criteria_best=EER_list[n_policy-self.start_iter+1]

        return EER_criteria_best/EER_real_best
    
    #----------------------------------------------------------------------------------------------
    # Criterios de seleccion
    #----------------------------------------------------------------------------------------------
    def truth_best_policy(self,elapsed_time):
        policy_id=self.df_test_estimates[(self.df_train['time_seconds']<=elapsed_time) & (self.df_train['time_seconds']>=self.start_time)]['truth'].idxmax()
        return policy_id

    def worst_policy(self,elapsed_time):
        policy_id=self.df_test_estimates[(self.df_train['time_seconds']<=elapsed_time) & (self.df_train['time_seconds']>=self.start_time)]['truth'].idxmin()
        return policy_id

    def last_policy(self,elapsed_time):
        '''Dado el tiempo transcurrido de aprendizaje, devuelve el indice de la ultima politica visitada.'''
        last_policy=self.df_train[self.df_train['time_seconds']<=elapsed_time]['n_policy'].max()
        return last_policy

    def best_policy_training(self,elapsed_time,n_traj_ep):
        # Lista de EER estimados con datos de train de la secunencia de politicas visitada hasta el momento
        estimated_EER_seq=self.df_train_estimates[(self.df_train['time_seconds']<=elapsed_time) & (self.df_train['time_seconds']>=self.start_time)][str(n_traj_ep)+'_traj_ep'].tolist()

        # Indice de la politica con mayor mean ER en train
        return estimated_EER_seq.index(max(estimated_EER_seq))+self.start_iter-1 # Debemos devolver el indice en la tabla y no en esta secuencia

    def best_policy_validation(self,elapsed_time,n_val_ep,freq):

        # Tiempos de validacion con frecuencia constante indicada
        current_val_times=[i for i in freq if i<=elapsed_time]

        # Indices de las politicas asociadas a esos tiempos, sus estimaciones de EER y el tiempo adicional consumido para su calculo
        current_val_policies=[]
        esti_time_seq=[]
        for time in current_val_times:
            policy_id=self.df_train.loc[(self.df_train['time_seconds']<=time) & (self.df_train['time_seconds']>=self.start_time)].index.max()

            current_val_policies.append(policy_id)  
            esti_time_seq.append(self.df_test_estimates[self.df_test_estimates['n_policy']==policy_id][str(n_val_ep)+'_val_ep'].values[0])

        esti_time_seq= [Converter.compress_decompress_list(i,compress=False) for i in esti_time_seq]

        # Dividir estimaciones de tiempos adicionales
        estimated_EER_seq=[]
        times_seq=[]
        for estimation, time in esti_time_seq:
            estimated_EER_seq.append(estimation)
            times_seq.append(time)

        # Indice de la politica (en la subsecuencia de las politicas asociadas a las frecuencias) con mayor mean ER en validacion
        indx_subseq=estimated_EER_seq.index(max(estimated_EER_seq))


        # Politica seleccionada y tiempo extra total invertido en su seleccion
        return current_val_policies[indx_subseq], sum(times_seq)
    
    #----------------------------------------------------------------------------------------------
    # Generadores de la evolucion completa de las metricas
    #----------------------------------------------------------------------------------------------
    def rank_evolution(self,x_times,n_ep=None,freq=None,criteria='last'):
        '''
        Genera lista de coordenadas OY (posiciones en el ranking real) a dibujar dependiendo del criterio seleccionado
        '''
        y_ranks=[]
        x_extras=[]
        val_time=0
        for time in x_times:
            if criteria=='truth_best':
                policy_id=self.truth_best_policy(time)
            if criteria=='worst':
                policy_id=self.worst_policy(time)
            if criteria=='last':
                policy_id=self.last_policy(time)
            if criteria=='best_train':
                policy_id=self.best_policy_training(time,n_ep)
            if criteria=='best_val':
                policy_id,val_time=self.best_policy_validation(time,n_ep,freq)
                x_extras.append(val_time)

            rank=self.real_ranking_position(time+val_time,policy_id)
            y_ranks.append(rank)

        if criteria=='best_val':
            return y_ranks, x_extras
        else:
            return y_ranks

    def magnitude_evolution(self,x_times,n_ep=None,freq=None,criteria='last',normalized=False,for_analyzer=False):
        '''
        Genera lista de coordenadas OY (diferencia en el EER truth con respecto al mejor real) a dibujar dependiendo del criterio seleccionado
        '''
        y_magnitudes=[]
        x_extras=[]
        val_time=0
        for time in x_times:
            if criteria=='truth_best':
                policy_id=self.truth_best_policy(time)
            if criteria=='worst':
                policy_id=self.worst_policy(time)
            if criteria=='last':
                policy_id=self.last_policy(time)
            if criteria=='best_train':
                policy_id=self.best_policy_training(time,n_ep)
            if criteria=='best_val':
                policy_id,val_time=self.best_policy_validation(time,n_ep,freq)
                x_extras.append(val_time)

            if for_analyzer:
                val_time=0
            magnitude=self.magnitude(time+val_time,policy_id,normalized=normalized)
            y_magnitudes.append(magnitude)

        if criteria=='best_val':
            return y_magnitudes, x_extras
        else:
            return y_magnitudes

    def degradation_evolution(self,x_times,global_metric,local_metric):
        return [self.degradation_level(time,global_metric,local_metric) for time in x_times]

    def effectiveness_evolution(self,x_times,n_ep=None,freq=None,criteria='last',normalized=False,for_analyzer=False):

        y_eff=[]
        x_extras=[]
        val_time=0
        for time in x_times:
            if criteria=='truth_best':
                policy_id=self.truth_best_policy(time)
            if criteria=='worst':
                policy_id=self.worst_policy(time)
            if criteria=='last':
                policy_id=self.last_policy(time)
            if criteria=='best_train':
                policy_id=self.best_policy_training(time,n_ep)
            if criteria=='best_val':
                policy_id,val_time=self.best_policy_validation(time,n_ep,freq)
                x_extras.append(val_time)

            if for_analyzer:
                val_time=0
            magnitude=self.effectiveness(time+val_time,policy_id,normalized=normalized)
            y_eff.append(magnitude)

        if criteria=='best_val':
            return y_eff, x_extras
        else:
            return y_eff
        
class EvolutionGrapher():
    '''
    Las funciones de esta clase permiten representar graficamente la evolucion de diferentes metricas.
    '''

    def __init__(self,algo,env,seed,resources,iter_max,perc_time_start=0.1,
                 list_n_traj_ep=[],
                 list_n_val_ep=[], list_n_val_freq=[]):
        
        # Primero generar los datos necesarios para las graficas
        if len(list_n_traj_ep)+len(list_n_val_ep)==0:
            Estimator.compute_estimates(algo,env,seed,resources,None,None)
        for n_ep in tqdm(list_n_traj_ep):
            Estimator.compute_estimates(algo,env,seed,resources,n_ep,'train')
        for n_ep in tqdm(list_n_val_ep):
            Estimator.compute_estimates(algo,env,seed,resources,n_ep,'test')

        # Guardar en una variable las 4 bases de datos
        self.generator=EvolutionGenerator(algo,env,seed,resources,perc_time_start)

        # Guardar el resto de variables
        self.algo=algo
        self.env=env
        self.seed=seed
        self.resources=resources
        self.list_n_traj_ep=list_n_traj_ep
        self.list_n_val_ep=list_n_val_ep
        self.list_n_val_freq=list_n_val_freq
        self.iter_max=iter_max
        self.start_iter=int(iter_max*perc_time_start)
        self.start_time=self.generator.df_train.loc[self.start_iter-1,'time_seconds']
  
    def graph_rank_evolution(self,only_literature_criteria=False):
        '''
        Dibuja en la misma grafica la evolucion de la posicion en el ranking real para cada posible criterio
        definido a partir de list_n_traj_ep, list_n_val_ep y list_n_val_freq, junto a las curvas truth, worst y last.
        '''

        fig=plt.figure(figsize=[7,5])
        plt.subplots_adjust(left=0.14,bottom=0.4,right=0.86,top=0.92,wspace=0.39,hspace=0.2)
        ax=plt.subplot(111)
        ax.grid(True, which='both',linestyle='--', linewidth=0.8,alpha=0.2)  
        colors=list(mcolors.TABLEAU_COLORS.keys())      

        x_times=Estimator.time_discretizer(self.algo,self.env,self.seed,self.resources,1,self.iter_max,self.start_time)

        if only_literature_criteria==False:
            # Dibujar la curva ideal
            y_truth_best=self.generator.rank_evolution(x_times,criteria='truth_best') 
            plt.plot(x_times, y_truth_best, linewidth=1,label='Ground truth',color=colors[0])

            # Dibujar la peor curva
            y_worst=self.generator.rank_evolution(x_times,criteria='worst') 
            plt.plot(x_times, y_worst, linewidth=1,label='Worst',color=colors[1])

        # Dibujar la curva de criterio last
        y_last=self.generator.rank_evolution(x_times,criteria='last') 
        plt.plot(x_times, y_last, linewidth=1,label='Last',color=colors[2])

        # Dibujar las curvas de criterio train
        i=3
        for n_ep in self.list_n_traj_ep:
            y_best=self.generator.rank_evolution(x_times,n_ep=n_ep,criteria='best_train') 
            plt.plot(x_times, y_best, linewidth=1,label=str(n_ep)+' train ep.',color=colors[i])
            i+=1
        
        # Dibujar las curvas de criterio test
        for n_ep in self.list_n_val_ep:
            for freq in self.list_n_val_freq:
                x_times_freq=Estimator.time_discretizer(self.algo,self.env,self.seed,self.resources,freq,self.iter_max,self.start_time)
                y_best,x_extra=self.generator.rank_evolution(x_times,n_ep=n_ep,freq=x_times_freq,criteria='best_val') 
                plt.plot([i+j for i,j in zip(x_times,x_extra)], y_best, linewidth=1,label=str(n_ep)+' val. ep. every '+str(freq)+' policies',color=colors[i])
                i+=1


        plt.title('Rank evolution')
        ax.set_xlabel("Total  time")
        ax.set_ylabel("Truth rank of the selected policy")
        plt.legend(title='Criteria',loc="upper center",bbox_to_anchor=(0.5, -0.3),ncol=3)

        if only_literature_criteria:
            plt.savefig('experiments_intuition/results/CriteriaComparison/figures/'+self.algo+'_'+self.env+'_seed'+str(self.seed)+'/rank_evolution_literature_critea.pdf')
        else:
            plt.savefig('experiments_intuition/results/CriteriaComparison/figures/'+self.algo+'_'+self.env+'_seed'+str(self.seed)+'/rank_evolution_all_criteria.pdf')

        #plt.show()
        plt.close()
  
    def graph_magnitude_evolution(self,only_literature_criteria=False, with_areas=False):
        '''
        Dibuja en la misma grafica la evolucion de la magnitud para cada posible criterio
        definido a partir de list_n_traj_ep, list_n_val_ep y list_n_val_freq, junto a las curvas truth, worst y last.
        '''
        def plot_magnitude_area(x_times,rows,colors,only_literature_criteria=False):
            '''
            Dibuja las areas bajo las curvas de evolucion de magnitud, con la magnitud normalizada.
            El onjetivo de esta grafica es poder distinguir bien cada criterio. El motivo del area es que nos
            interesa minimizar la magnitud para cualquiera que sea el tiempo de aprendizaje.
            '''

            if only_literature_criteria==False:
                # Ground truth
                ax=plt.subplot(rows,2,2)
                y_trurh_best=self.generator.magnitude_evolution(x_times,criteria='truth_best',normalized=True) 
                plt.plot(x_times, y_trurh_best, linewidth=1,label='Ground truth',color=colors[0])
                plt.fill_between( x_times,y_trurh_best,[0]*len(x_times),alpha=0.5,color=colors[0])
                plt.ylim([-0.1,1.1])
                plt.xticks([])
                plt.title('Normalized magnitude area')
                
                # Worst
                ax=plt.subplot(rows,2,4)
                y_worst=self.generator.magnitude_evolution(x_times,criteria='worst',normalized=True) 
                plt.plot(x_times, y_worst, linewidth=1,label='Worst',color=colors[1])
                plt.fill_between(x_times,y_worst,[0]*len(x_times),alpha=0.5,color=colors[1])
                plt.ylim([-0.1,1.1])
                plt.xticks([])

            # Last
            if only_literature_criteria:
                ax=plt.subplot(rows,2,2)
                plt.title('Normalized magnitude area')
                j=2
            else:
                ax=plt.subplot(rows,2,6)
            y_last=self.generator.magnitude_evolution(x_times,criteria='last',normalized=True) 
            plt.plot(x_times, y_last, linewidth=1,label='Last',color=colors[2])
            plt.fill_between(x_times,y_last,[0]*len(x_times),alpha=0.5,color=colors[2])
            plt.ylim([-0.1,1.1])
            plt.xticks([])

            # Best train
            i=4
            for n_ep in self.list_n_traj_ep:
                if only_literature_criteria:
                    ax=plt.subplot(rows,2,j*2)
                    j+=1
                else:
                    ax=plt.subplot(rows,2,i*2)
                y_best=self.generator.magnitude_evolution(x_times,n_ep=n_ep,criteria='best_train',normalized=True) 
                plt.plot(x_times, y_best, linewidth=1,label=str(n_ep)+' train ep.',color=colors[i-1])
                plt.fill_between(x_times,y_best,[0]*len(x_times),alpha=0.5,color=colors[i-1])
                plt.ylim([-0.1,1.1])
                plt.xticks([])
                i+=1

            # Best test
            for n_ep in self.list_n_val_ep:
                for freq in self.list_n_val_freq:
                    if only_literature_criteria:
                        ax=plt.subplot(rows,2,j*2)
                        j+=1
                    else:
                        ax=plt.subplot(rows,2,i*2)
                    x_times_freq=Estimator.time_discretizer(self.algo,self.env,self.seed,self.resources,freq,self.iter_max,self.start_time)
                    y_best,x_extra=self.generator.magnitude_evolution(x_times,n_ep=n_ep,freq=x_times_freq,criteria='best_val',normalized=True) 
                    plt.plot([i+j for i,j in zip(x_times,x_extra)], y_best, linewidth=1,label=str(n_ep)+' val. ep. every '+str(freq)+' policies',color=colors[i-1])
                    plt.fill_between([i+j for i,j in zip(x_times,x_extra)],y_best,[0]*len(x_times),alpha=0.5,color=colors[i-1])
                    plt.ylim([-0.1,1.1])
                    if i<rows:
                        plt.xticks([])

                    i+=1

            ax.set_xlabel("Elapsed total time")

        if only_literature_criteria==False:
            rows=3+len(self.list_n_traj_ep)+len(self.list_n_val_ep)*len(self.list_n_val_freq)
            fig=plt.figure(figsize=[7,5])
            plt.subplots_adjust(left=0.14,bottom=0.4,right=0.86,top=0.92,wspace=0.39,hspace=0.2)
            
        else:
            rows=1+len(self.list_n_traj_ep)+len(self.list_n_val_ep)*len(self.list_n_val_freq)
            fig,axes=plt.subplots(rows,2,figsize=[10,5],gridspec_kw={'width_ratios': [2,1]})
            plt.subplots_adjust(left=0.1,bottom=0.4,right=0.97,top=0.92,wspace=0.09,hspace=0.1)
            for i in range(rows): # Oculatr ejes de las subgraficas de la primera columna
                fig.delaxes(axes[i, 0])
            
        colors=list(mcolors.TABLEAU_COLORS.keys())

        # GRAFICA 1: Curvas
        if with_areas:
            ax=plt.subplot2grid((rows, 2), (0, 0), rowspan=rows)
        else:
            ax=plt.subplot(111)
        ax.grid(True, which='both',linestyle='--', linewidth=0.8,alpha=0.2)     

        x_times=Estimator.time_discretizer(self.algo,self.env,self.seed,self.resources,1,self.iter_max,self.start_time)

        if only_literature_criteria==False:
            # Dibujar la curva ideal
            y_trurh_best=self.generator.magnitude_evolution(x_times,criteria='truth_best') 
            plt.plot(x_times, y_trurh_best, linewidth=1,label='Ground truth',color=colors[0])

            # Dibujar la peor curva
            y_worst=self.generator.magnitude_evolution(x_times,criteria='worst') 
            plt.plot(x_times, y_worst, linewidth=1,label='Worst',color=colors[1])

        # Dibujar la curva de criterio last
        y_last=self.generator.magnitude_evolution(x_times,criteria='last') 
        plt.plot(x_times, y_last, linewidth=1,label='Last',color=colors[2])

        # Dibujar las curvas de criterio train
        i=3
        for n_ep in self.list_n_traj_ep:
            y_best=self.generator.magnitude_evolution(x_times,n_ep=n_ep,criteria='best_train') 
            plt.plot(x_times, y_best, linewidth=1,label=str(n_ep)+' train ep.',color=colors[i])
            i+=1
        
        # Dibujar las curvas de criterio test
        for n_ep in self.list_n_val_ep:
            for freq in self.list_n_val_freq:
                x_times_freq=Estimator.time_discretizer(self.algo,self.env,self.seed,self.resources,freq,self.iter_max,self.start_time)
                y_best,x_extra=self.generator.magnitude_evolution(x_times,n_ep=n_ep,freq=x_times_freq,criteria='best_val') 
                plt.plot([i+j for i,j in zip(x_times,x_extra)], y_best, linewidth=1,label=str(n_ep)+' test ep. ; '+str(freq)+' freq.',color=colors[i])
                i+=1
        plt.title('Magnitude evolution')
        ax.set_xlabel("Elapsed total  time")
        ax.set_ylabel("Difference of truth\nexpected episodic reward\nbetween selected policy and real best")
        plt.legend(title='Criteria',loc="upper center",bbox_to_anchor=(0.5, -0.3),ncol=3)

        # GRAFICA 2: Areas
        if with_areas:  
            plot_magnitude_area(x_times,rows,colors,only_literature_criteria)

        if only_literature_criteria and with_areas:
            plt.savefig('experiments_intuition/results/CriteriaComparison/figures/'+self.algo+'_'+self.env+'_seed'+str(self.seed)+'/magnitude_evolution_literature_criteria.pdf')
        else:
            plt.savefig('experiments_intuition/results/CriteriaComparison/figures/'+self.algo+'_'+self.env+'_seed'+str(self.seed)+'/magnitude_evolution_all_criteria.pdf')

        #plt.show()
        plt.close()

    def learning_curve(self):
        '''
        Evolucion del proceso de apendizaje usando el mean episodic reward que representa el groun truth.
        '''

        fig=plt.figure(figsize=[15,2.5])
        plt.subplots_adjust(left=0.1,bottom=0.2,right=0.94,top=0.82,wspace=0.21,hspace=0.2)
        ax=plt.subplot(111)
        ax.grid(True, which='both',linestyle='--', linewidth=0.8,alpha=0.2)

        # Dibujar curva sin test durante el proceso (ultima politica observada)
        plt.plot(self.generator.df_train['time_seconds'][:self.iter_max], self.generator.df_test_estimates['truth'], linewidth=1,color='black')

        plt.title('Learning-curve')
        ax.set_xlabel("Elapsed learning time")
        ax.set_ylabel("Truth expected episodic reward\nof the last policy")
        plt.savefig('experiments_intuition/results/CriteriaComparison/figures/'+self.algo+'_'+self.env+'_seed'+str(self.seed)+'/learning_curve.pdf')
        #plt.show()
        plt.close()

    def graph_degradation_evolution(self,global_metric,local_metric):
        '''
        Esta funcion es la version mejorada/completa de la anterior. Ademas de la la learning-curve (evolucion del truth expected episodic reward),
        tambien incluye evolucion de nivel de degradacion (global), de degradacion por actualizacion (local), y analisis individual comparativo
        de degradacion y truth en diferentes casos.
        '''

        fig, axes = plt.subplots(4,4,figsize=(12,5),gridspec_kw={'height_ratios': [1,3,1,3]})
        plt.subplots_adjust(left=0.08,bottom=0.25,right=0.8,top=0.92,wspace=0.39,hspace=0.1)
        list_axes=[ax for _,ax in enumerate(axes.flat)]
        for i in range(12):
            list_axes[i].set_visible(False)

        def KDE_consecutive_policies(n_policy,ax,title,local_metric,metrics_to_show,first=False):
            
            # Obtener los ep. reward truth normalizados de las dos politicas de interes
            column_ep_rewards=[Converter.compress_decompress_list(i,compress=False)[:500] for i in self.generator.df_test['ep_rewards']]
            prev_ER=column_ep_rewards[n_policy-1][:500]
            current_ER=column_ep_rewards[n_policy][:500]
            max_ep_rw=max(prev_ER+current_ER)
            min_ep_rw=min(prev_ER+current_ER)
            norm_prev_ER=Converter.normalize_list(prev_ER,min_ep_rw,max_ep_rw)
            norm_current_ER=Converter.normalize_list(current_ER,min_ep_rw,max_ep_rw)

            
            # KDE a partir de las dos muestras anteriores
            if local_metric=='greater_prob':
                x_d=np.linspace(0,1, 1000)
                avg_ep_rw=[]
                x_position=[0]
                for sample in [norm_prev_ER,norm_current_ER]:
                    x=np.array(sample)
                    kde = KernelDensity(bandwidth=0.05, kernel='gaussian')
                    kde.fit(x[:, None])
                    y_prob = np.exp(kde.score_samples(x_d[:, None]))

                    ax.plot(np.full_like(x, -0.01)+x_position[-1],x, '_', markeredgewidth=1,color='grey',alpha=0.5)
                    ax.fill_betweenx( x_d,y_prob+x_position[-1],min(y_prob)+x_position[-1],color='red',alpha=0.5)

                    x_position.append(max(y_prob)*2)
                    avg_ep_rw.append(np.mean(x))

                ax.plot(x_position[:2],avg_ep_rw,color='black')
                ax.set_xlabel(title+'\n$P(X_{i-1}>X_i)=$'+str(format(metrics_to_show[0], ".2e"))+'\n$\delta_{i-1,i}=$'+str(format(metrics_to_show[1], ".2e"))+'\n$\Delta f^-_i=$'+str(format(metrics_to_show[2], ".2e")))
                ax.set_xticks(ticks=x_position[:2], labels=[str(n_policy-1),str(n_policy)])
                if first:
                    ax.set_ylabel('Normalized\nepisodic reward')

            if local_metric in ['paired_diff_probpos_meanpos','paired_diff_median','paired_diff_probpos']:
                x_d=np.linspace(-1,1, 1000)
                x=np.array(norm_prev_ER)-np.array(norm_current_ER)
                kde = KernelDensity(bandwidth=0.05, kernel='gaussian')
                kde.fit(x[:, None])
                y_prob = np.exp(kde.score_samples(x_d[:, None]))

                ax.plot(x,np.full_like(x, -0.01), '|', markeredgewidth=1,color='grey',alpha=0.5)
                ax.plot(np.mean(norm_prev_ER)-np.mean(norm_current_ER),-0.01,'|',markeredgewidth=2,color='black',label='diff mean')
                ax.plot(np.mean([i for i in x if i>0]),-0.01,'|', color='blue',markeredgewidth=2,label='+diff mean')
                ax.plot(np.median(x),-0.01,'|', color='green',markeredgewidth=2,label='diff median')
                ax.fill_between( x_d,y_prob,min(y_prob),color='red',alpha=0.5)
                ax.axvline(x=0, color='black', linestyle='--',linewidth=0.5)
                perc_pos=int(len([i for i in x if i>=0])*100)/len(x)
                perc_neg=int(len([i for i in x if i<0])*100)/len(x)
                
                ax.set_xlabel(str(perc_neg)+'%    '+str(perc_pos)+'%\n\n'+title+'\n$\delta_{i-1,i}=$'+str(format(metrics_to_show[1], ".2e"))+'\n$\Delta f^-_i=$'+str(format(metrics_to_show[2], ".2e")))
                ax.legend(fontsize=5)
                if first:
                    ax.set_ylabel('Normalized\nepisodic reward\npaired diff. distribution')


        # Evolucion del nivel de degradacion
        ax=plt.subplot(4,4,(1,4))
        degradation_level=[0 for _ in range(int(self.start_iter))]+self.generator.degradation_evolution(self.generator.df_train['time_seconds'].tolist()[self.start_iter:self.iter_max],global_metric,local_metric)
        ax.imshow(np.array(degradation_level).reshape(1, -1), cmap="gray_r", aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Degradation evolution (global: '+global_metric+'; local: '+local_metric+')')

        # Learning-curve con lineas verticales que representan los update degradations
        ax=plt.subplot(4,4,(5,8))
        ax.grid(True, which='both',linestyle='--', linewidth=0.8,alpha=0.2)

        cmap_white_to_red = mcolors.LinearSegmentedColormap.from_list("white_red", ["white", "red"])
        for i in range(self.iter_max):
            cmap=Converter.generate_colormap(self.generator.df_test_estimates['update_deg_'+local_metric][i],cmap_white_to_red,0,1)
            ax.axvline(x=i, color=cmap, linestyle='-', linewidth=1) 

        plt.plot(list(range(self.iter_max))[:int(self.iter_max*.1)], self.generator.df_test_estimates['truth'][:int(self.iter_max*.1)], linewidth=.5,linestyle='--',color='black')
        plt.plot(list(range(self.iter_max))[int(self.iter_max*.1)-1:], self.generator.df_test_estimates['truth'][int(self.iter_max*.1)-1:], linewidth=1,color='black')
        plt.xlim([0,self.iter_max])
        ax.set_ylabel('Episodic reward')
        ax.set_xlabel('Learning iteration ($i$)')

        # KDE de iteracion actual y previa para comparar similitud entre metrica de degradacion y estimacion truth
        update_degradation=abs(np.array(self.generator.df_test_estimates['update_deg_'+local_metric][self.start_iter:])) # abs porque al hacer np.array los ceros salen -0.0

        if local_metric not in ['relative_reward_diff']:
            if local_metric=='greater_prob':
                dominances=np.array(self.generator.df_test_estimates['update_dominances'][self.start_iter:])
            else:
                dominances=[None]*(self.iter_max-self.start_iter)

            truth_degradation=[0]+[self.generator.df_test_estimates['truth'][i-1]-self.generator.df_test_estimates['truth'][i] for i in range(1,self.iter_max)]
            truth_degradation=abs(Converter.normalize_list([i if i>0 else 0 for i in truth_degradation]))[self.start_iter:]

            true_negative=['True negative',min((i for i, deg in enumerate(truth_degradation) if deg==0),key=lambda i: update_degradation[i],default=None)]
            true_positive=['True positive',max((i for i, deg in enumerate(truth_degradation) if deg>0),key=lambda i: update_degradation[i],default=None)]
            false_negative=['False negative',max((i for i, deg in enumerate(truth_degradation) if deg==0),key=lambda i: update_degradation[i],default=None)]
            false_positive=['False positive',max((i for i, deg in enumerate(update_degradation) if deg==0),key=lambda i: truth_degradation[i],default=None)]

            highlighted_iter=[true_negative,true_positive,false_negative,false_positive]
            for i in range(4):
                ax=plt.subplot(4,4,13+i) 
                if highlighted_iter[i][1]!=None:
                    metrics_to_show=[dominances[highlighted_iter[i][1]],update_degradation[highlighted_iter[i][1]],truth_degradation[highlighted_iter[i][1]]]
                    KDE_consecutive_policies(highlighted_iter[i][1]+self.start_iter,ax,highlighted_iter[i][0]+' ($i$='+str(highlighted_iter[i][1]+self.start_iter)+')',local_metric,metrics_to_show,i==0)
                else:
                    ax.set_xlabel('No '+highlighted_iter[i][0])
                    ax.set_frame_on(False)  
                    ax.set_xticks([])  
                    ax.set_yticks([])  
                    ax.set_xticklabels([]) 
                    ax.set_yticklabels([])  
        else:
            for i in range(4):
                plt.subplot(4,4,13+i).axis('off')
        # Barras de colores
        Converter.generate_colorbar(fig,[0.84, 0.55, 0.015, 0.4],cmap_white_to_red,[0,max(update_degradation)],'Update degradation ($\delta_{i-1,i}$)')
        Converter.generate_colorbar(fig,[0.92, 0.55, 0.015, 0.4],'Greys',[0,max(degradation_level)],'Degradation level ($\delta_i$)')

        plt.savefig('experiments_intuition/results/CriteriaComparison/figures/'+self.algo+'_'+self.env+'_seed'+str(self.seed)+'/degradation_evolution_'+global_metric+'_'+local_metric+'.pdf')
        #plt.show()
        plt.close()

    def MAEB_graph_degradation_evolution(self,global_metric,local_metric,lim_max):
        '''
        Evalucion de truth expected episodic reward con algunas degradaciones de algunos instantes.
        '''

        plt.rc('font', family='serif',size=18)
        plt.rc('text', usetex=True)

        fig, ax = plt.subplots(1,1,figsize=(7,4))
        plt.subplots_adjust(left=0.15,bottom=0.25,right=0.8,top=0.92,wspace=0.39,hspace=0.1)
 
        # Evolucion del nivel de degradacion
        degradation_level=[0 for _ in range(int(self.start_iter))]+self.generator.degradation_evolution(self.generator.df_train['time_seconds'].tolist()[self.start_iter:self.iter_max],global_metric,local_metric)

        for i in [4,39,15,25,29,46]:
            print(degradation_level[int(self.iter_max*.1)-1:int(self.iter_max*.1)-1+lim_max][i])

        ax.axvline(x=4, color='red', linestyle='-', linewidth=1) 
        ax.axvline(x=39, color='red', linestyle='-', linewidth=1) 
        ax.axvline(x=15, color='red', linestyle='-', linewidth=1) 
        ax.axvline(x=25, color='red', linestyle='-', linewidth=1) 
        ax.axvline(x=29, color='red', linestyle='-', linewidth=1) 
        ax.axvline(x=46, color='red', linestyle='-', linewidth=1) 
        plt.plot(list(range(0,lim_max)), self.generator.df_test_estimates['truth'][int(self.iter_max*.1)-1:int(self.iter_max*.1)-1+lim_max], linewidth=1,color='black')

        ax.set_ylabel('Recompensa real')
        ax.set_xlabel('Tiempo de aprendizaje ($t$)')

        plt.savefig('experiments_intuition/results/MAEB/deg_evolution.pdf')
        #plt.show()
        plt.close()

class EstimationAnalyzer():  
    '''
    Las funciones de esta clase permiten analizar graficamente el coste de las estimaciones de expected episodic reward
    a partir de los datos test, y la precision de seleccion de las estimaciones con test y train. Los analisis estan enfocados
    en un unico proceso determinado por una combinacion de algortimo-environment-semilla-resursos
    '''  

    def __init__(self,algo,env,seed,resources,
                 list_n_traj_ep,
                 list_n_val_ep,
                 iter_max,perc_time_start=0.1):
        
        # Primero generar los datos necesarios para las graficas
        for n_ep in tqdm(list_n_traj_ep):
            Estimator.compute_estimates(algo,env,seed,resources,n_ep,'train')
        for n_ep in tqdm(list_n_val_ep):
            Estimator.compute_estimates(algo,env,seed,resources,n_ep,'test')

        # Guardar en una variable las 4 bases de datos
        self.generator=EvolutionGenerator(algo,env,seed,resources,perc_time_start)

        # Leer o generar un csv en  donde se guaradara el valor mas grande para n_ep que consume menos o igual que l 25% del timepo disponible en validacion
        path_csv='experiments_intuition/results/CriteriaComparison/data/test_affordable_n_ep_by_process.csv'
        if not os.path.exists(path_csv):
            df = pd.DataFrame(columns=['process_id','n_ep_0.25'])
            df.to_csv(path_csv, index=False)

        # Guardar el resto de variables
        self.algo=algo
        self.env=env
        self.seed=seed
        self.resources=resources
        self.list_n_traj_ep=list_n_traj_ep
        self.list_n_val_ep=list_n_val_ep
        self.iter_max=iter_max 
        self.start_iter=int(iter_max*perc_time_start)
        self.start_time=self.generator.df_train.loc[self.start_iter-1,'time_seconds']
        self.path_csv_affordable_n_ep_by_process=path_csv

    #----------------------------------------------------------------------------------------------
    # Calculo de matrices numericas
    #----------------------------------------------------------------------------------------------
    def matrix_for_cost_analysis(self):
        '''
        Genera matriz numerica con los datos necesarios para conocer por politica de la secuencia total: 
        - Los porcentages de coste de validacion frente al coste de iteracion train para diferentes numeros de episodios de validacion
        - Tiempo consumido por iteracion train (tiempo de actualizacion+tiempo de interaccion)
        - Numero de episodios por trajectorias almacenadas en cada iteracion train
        - Caracterizacion de cada politica usando datos truth: mean episodic reward (goodness), var episodic reward (variability) y mean episode lenght (durability)
        '''

        matrix=[]

        # Lista de numero de inicializaciones de episodio por iteracion
        num_ep_init_per_iter=[np.sum(Converter.compress_decompress_list(i,compress=False)) for i in self.generator.df_train['traj_ep_end'][:self.iter_max]]
        matrix.append(Converter.normalize_list(num_ep_init_per_iter))

        # Lista de tiempos de interaccion por iteracion
        matrix.append([self.generator.df_train['time_seconds'][0]]+[self.generator.df_train['time_seconds'][i+1]-self.generator.df_train['time_seconds'][i] for i in range(self.iter_max-1)])

        # Listas de tiempos de validacion con n_ep fijo por iteracion
        for n_ep in self.list_n_val_ep:
            matrix.append([Converter.compress_decompress_list(i,compress=False)[1] for i in self.generator.df_test_estimates[str(n_ep)+'_val_ep'][:self.iter_max]])

        # Porcentage de filas de validacion con respecto fila de interaccion
        for_affordable_n_ep=[]
        for i in range(2,len(self.list_n_val_ep)+2):
            matrix[i]=np.array(matrix[i])/np.array(matrix[1])
            for_affordable_n_ep.append(matrix[i][self.start_iter-1:])

        # Guardar minimo entre los maximos valores de list_n_val_ep por politica que tienen un porcentage de validacion menor que 0.25
        for_affordable_n_ep=np.array(for_affordable_n_ep).T
        max_n_ep_by_policy=[next((self.list_n_val_ep[i] for i, perc in enumerate(percs_n_ep) if perc <= 0.25), 0) for percs_n_ep in for_affordable_n_ep]
        with open(self.path_csv_affordable_n_ep_by_process, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([self.algo+'_'+self.env+'_seed'+str(self.seed)]+[int(np.mean(max_n_ep_by_policy))]) 

        # Normalizar tiempos de interaccion
        matrix[1]=Converter.normalize_list(matrix[1])

        # Listas para las identificar el tipo de politica (buena/mala, estocastica/determinista, duradera/bolatil)
        goodness=Converter.normalize_list(self.generator.df_test_estimates['truth'][:self.iter_max])
        variability= Converter.normalize_list([np.var(Converter.compress_decompress_list(i,compress=False)[:500]) for i in self.generator.df_test['ep_rewards'][:self.iter_max]])
        durability=Converter.normalize_list([np.mean(Converter.compress_decompress_list(i,compress=False)[:500]) for i in self.generator.df_test['ep_lens'][:self.iter_max]])
        matrix=np.vstack((np.array([goodness,variability,durability]),np.array(matrix)))

        return matrix
    
    def matrix_for_accuracy_analysis(self,freq=None,train_or_test='train',global_deg_metric='mean_update_deg',local_deg_metric='greater_prob'):
        '''
        Genera matriz numerica con los datos necesarios para interpretar como de buenas son las estimaciones 
        para seleccionar la mejor politica de la secuencia actual (no se considera el tiempo invertido en validacion,
        solo es para comparar la precision de seleccion de los estimadores): 
        - Nivel de degradacion de las diferentes secuencias (considerando cada vez una iteracion mas).
        Con esta medimetrica se pretende representar la "dificultad" de seleccionar la mejor politica. Cuanto mayor degradacion, 
        mas sencilla su distincion, aunque mayor magnitud tendran los errores de seleccion.
        - Evoluciones de magnitud en la propia secuncia (sin considerar el extra de tiempo invertido en validacion).
        '''
        matrix=[]

        # Lista de niveles de degradacion
        degradation_level=self.generator.degradation_evolution(self.generator.df_train['time_seconds'].tolist()[self.start_iter-1:self.iter_max],global_deg_metric,local_deg_metric)
        matrix.append(degradation_level)

        # Listas de normalized magnitude para cada posible n_ep
        if train_or_test=='train':
            list_n_ep=self.list_n_traj_ep
            criteria='best_train'
        if train_or_test=='test':
            criteria='best_val'
            list_n_ep=self.list_n_val_ep
            freq=Estimator.time_discretizer(self.algo,self.env,self.seed,self.resources,freq,self.iter_max,self.start_time)

        for n_ep in list_n_ep:
            if train_or_test=='train':
                magnitude=self.generator.magnitude_evolution(self.generator.df_train['time_seconds'][self.start_iter-1:self.iter_max],n_ep,freq,criteria,normalized=True,for_analyzer=True)
            if train_or_test=='test':
                magnitude,extra_time=self.generator.magnitude_evolution(self.generator.df_train['time_seconds'][self.start_iter-1:self.iter_max],n_ep,freq,criteria,normalized=True,for_analyzer=True)
            matrix.append(magnitude)

        return np.array(matrix)
    
    def matrix_for_train_vs_test_analysis(self):
        matrix=[]

        # Numero de episodios en cada iteracion train
        traj_ep_ends=[Converter.compress_decompress_list(i,compress=False) for i in self.generator.df_train['traj_ep_end'][:self.iter_max]]
        ep_ends_per_iter=[sum(chain.from_iterable(i)) for i in traj_ep_ends]
        matrix.append(ep_ends_per_iter)

        # Evolucion de truth
        matrix.append(Converter.normalize_list(self.generator.df_test_estimates['truth'][:self.iter_max]))

        # Evolucion de estimaciones con train
        for n_ep in self.list_n_traj_ep:
            matrix.append(Converter.normalize_list(self.generator.df_train_estimates[str(n_ep)+'_traj_ep'][:self.iter_max]))

        # Evolucion de los loss para la actualizacion en el learning
        matrix.append(Converter.normalize_list(self.generator.df_train['policy_loss'][:self.iter_max]))

        # Evolucion de estimaciones con  test
        for n_ep in self.list_n_val_ep:
            matrix.append(Converter.normalize_list([Converter.compress_decompress_list(i,compress=False)[0] for i in self.generator.df_test_estimates[str(n_ep)+'_val_ep']]))
       
        return np.array(matrix)
    #----------------------------------------------------------------------------------------------
    # Calculo de matrices de colores
    #----------------------------------------------------------------------------------------------
    def colored_matrix_for_cost_analysis(self,matrix,cost_perc_threshold=0.25):
        """Crea una imagen coloreada basada en la matriz para el analisis de los costes de estimacion"""
        rows, cols = matrix.shape
        colored_matrix = np.zeros((rows, cols, 3))  # Matriz para colores RGB
        
        # Filas iniciales (escala azul): goodness, variability y durability de la politica
        for i in range(3):
            for j in range(cols):
                blue_value = matrix[i, j]  # Valor entre 0 y 1
                colored_matrix[i, j] = Converter.generate_colormap(blue_value, 'Blues', 0, 1)[:3]  # Usar azul para estas filas
        
        # Fila 4,5 (escala de grises): tiempo por iteracion train
        for i in range(3,5):
            for j in range(cols):
                gray_value = 1 - matrix[i, j]  # Invertir el valor para la escala de grises
                colored_matrix[i, j] = [gray_value, gray_value, gray_value]
        
        # Filas restantes (escala verde y roja): porcentages de tiempo de validacion frente al tiempo por iteracion train
        for i in range(5, rows):
            for j in range(cols):
                value = matrix[i, j]
                if value <= cost_perc_threshold:
                    # Usar la escala verde invertida (ahora 0 es más oscuro y 1 es más claro)
                    colored_matrix[i, j] = Converter.generate_colormap(cost_perc_threshold - value, 'Greens', 0, cost_perc_threshold)[:3]
                else:
                    if value>1: # Cortar el porcentage en 1
                        value=1
                    colored_matrix[i, j] = Converter.generate_colormap(value, 'Reds', cost_perc_threshold, 1)[:3]
        
        return colored_matrix
    
    def colored_matrix_for_accuracy_analysis(self,matrix):
        """Crea una imagen coloreada basada en la matriz para el analisis de la precision de seleccion de los estimadores"""
        rows, cols = matrix.shape
        colored_matrix = np.zeros((rows, cols, 3))  # Matriz para colores RGB
        
        # Fila inicial (escala azul): varianza del truth entre el mejor 25% de las politicas de la secunecia
        for j in range(cols):
            blue_value = matrix[0, j]  # Valor entre 0 y 1
            colored_matrix[0, j] = Converter.generate_colormap(blue_value, 'Blues', 0, max(matrix[0, :]))[:3]  # Usar azul para estas filas
        
        # Resto de filas (escala de grises): magnitud normalizada
        for i in range(1,rows):
            for j in range(cols):
                gray_value = 1 - matrix[i, j]  # Invertir el valor para la escala de grises
                colored_matrix[i, j] = [gray_value, gray_value, gray_value]
        
        return colored_matrix
    
    def colored_matrix_for_train_vs_test_analysis(self,matrix):
        rows, cols = matrix.shape
        colored_matrix = np.zeros((rows, cols, 3))  # Matriz para colores RGB
            
        # Filas de evolucion de estimaciones train
        for i in range(matrix.shape[0]):
            for j in range(cols):
                gray_value = 1 - matrix[i, j]  # Invertir el valor para la escala de grises
                colored_matrix[i, j] = [gray_value, gray_value, gray_value]

        return colored_matrix
    
    def MAEB_colored_matrix_for_cost_analysis(self,matrix,cost_perc_threshold=0.25):
        """Crea una imagen coloreada basada en la matriz para el analisis de los costes de estimacion"""
        rows, cols = matrix.shape
        colored_matrix = np.zeros((rows, cols, 3))  # Matriz para colores RGB
        
        # Filas iniciales (escala azul): goodness, variability y durability de la politica
        for i in range(3):
            for j in range(cols):
                gray_value = 1 - matrix[i, j]  # Invertir el valor para la escala de grises
                colored_matrix[i, j] = [gray_value, gray_value, gray_value]
        
        # Fila 4,5 (escala de grises): tiempo por iteracion train
        for i in range(3,5):
            for j in range(cols):
                gray_value = 1 - matrix[i, j]  # Invertir el valor para la escala de grises
                colored_matrix[i, j] = [gray_value, gray_value, gray_value]
        
        # Filas restantes (escala verde y roja): porcentages de tiempo de validacion frente al tiempo por iteracion train
        for i in range(5, rows):
            for j in range(cols):
                value = matrix[i, j]
                if value <= cost_perc_threshold:
                    # Usar la escala verde invertida (ahora 0 es más oscuro y 1 es más claro)
                    colored_matrix[i, j] = Converter.generate_colormap(cost_perc_threshold - value, 'Greens', 0, cost_perc_threshold)[:3]
                else:
                    if value>1: # Cortar el porcentage en 1
                        value=1
                    colored_matrix[i, j] = Converter.generate_colormap(value, 'Reds', cost_perc_threshold, 1)[:3]
        
        return colored_matrix
    
    #----------------------------------------------------------------------------------------------
    # Graficas para el analisis de coste de estimaciones y precison de seleccion de estimaciones
    #----------------------------------------------------------------------------------------------
    def graph_cost_analysis(self):
        
        # Genara la matriz numerica a partir de los datos
        matrix=self.matrix_for_cost_analysis()

        # Generar matriz de colores
        colored_matrix = self.colored_matrix_for_cost_analysis(matrix)

        fig, ax = plt.subplots(figsize=(20, 6))
        plt.subplots_adjust(left=0.09, bottom=0.2, right=0.81, top=0.82, wspace=0.39, hspace=0.2)
        im = ax.imshow(colored_matrix, aspect='auto')

        # Crear barras de colores
        Converter.generate_colorbar(fig,[0.82, 0.15, 0.015, 0.7],'Blues',[0,1],'Policy characteristic level')
        Converter.generate_colorbar(fig,[0.87, 0.15, 0.015, 0.7],'gray_r',[0,1],'Normalized train data')
        Converter.generate_colorbar(fig,[0.92, 0.15, 0.015, 0.7],'Greens_r',[0,.25],'')
        Converter.generate_colorbar(fig,[0.95, 0.15, 0.015, 0.7],'Reds',[.25,1],'Percentage of iteration time (clipped in 1)')
               
        # Etiquetas para los ejes
        row_labels = ['Goodness', 'Variability', 'Durability','Train iter. ep. initis','Train iter. time']+[str(i)+' val. ep.' for i in self.list_n_val_ep]
        ax.set_yticks(np.arange(matrix.shape[0]))  # Establece las posiciones de las filas
        ax.set_yticklabels(row_labels,fontsize=12)  # Establece las etiquetas de las filas
        ax.set_xlabel("Policy in sequence", fontsize=14)
        
        plt.savefig('experiments_intuition/results/CriteriaComparison/figures/'+self.algo+'_'+self.env+'_seed'+str(self.seed)+'/estimation_cost_analysis.pdf')
        #plt.show()
        plt.close()

    def graph_accuracy_analysis(self,train_or_test='train'):

        # Genara la matriz numerica a partir de los datos
        if train_or_test=='train':
            freq=None
        if train_or_test=='test':
            freq=1
        matrix=self.matrix_for_accuracy_analysis(freq,train_or_test)

        # Generar matriz de colores
        colored_matrix = self.colored_matrix_for_accuracy_analysis(matrix)

        fig, ax = plt.subplots(figsize=(20, 6))
        plt.subplots_adjust(left=0.076, bottom=0.2, right=0.81, top=0.82, wspace=0.39, hspace=0.2)
        im = ax.imshow(colored_matrix, aspect='auto')

        # Crear barras de color
        Converter.generate_colorbar(fig,[0.82, 0.15, 0.015, 0.7],'Blues',[0,max(matrix[0])],'Degradation level') 
        Converter.generate_colorbar(fig,[0.87, 0.15, 0.015, 0.7],'gray_r',[0,1],'Normalized magnitude') 
       
        # Etiquetas para los ejes
        if train_or_test=='train':
            row_labels = ['Degradation']+[str(i)+' traj. ep.' for i in self.list_n_traj_ep]
        if train_or_test=='test':
            row_labels = ['Degradation']+[str(i)+' val. ep.' for i in self.list_n_val_ep]
        ax.set_yticks(np.arange(matrix.shape[0]))  # Establece las posiciones de las filas
        ax.set_yticklabels(row_labels,fontsize=12)  # Establece las etiquetas de las filas
        ax.set_xticks(list(range(0,matrix.shape[1],50)))  
        ax.set_xticklabels(list(range(self.start_iter,self.iter_max,50)),fontsize=12)
        ax.set_xlabel("Number of iterations", fontsize=14)
        
        plt.savefig('experiments_intuition/results/CriteriaComparison/figures/'+self.algo+'_'+self.env+'_seed'+str(self.seed)+'/estimation_accuracy_analysis_'+train_or_test+'.pdf')
        #plt.show()
        plt.close()

    def graph_train_vs_test_estimates(self):
        # Genara la matriz numerica a partir de los datos
        matrix=self.matrix_for_train_vs_test_analysis()
        # Generar matriz de colores
        colored_matrix = self.colored_matrix_for_train_vs_test_analysis(matrix[1:])


        # Dibujar graficas
        fig, axes = plt.subplots(3,2,figsize=(25,7),gridspec_kw={'height_ratios': [1, 3,3],'width_ratios': [1,1]})
        plt.subplots_adjust(left=0.07, bottom=0.2, right=0.98, top=0.94, wspace=0.05, hspace=0.08)
        list_axes=[ax for _,ax in enumerate(axes.flat)]

        ax=list_axes[0]
        ax.bar(list(range(len(matrix[0]))), matrix[0], color='grey', edgecolor=None)
        for n_ep in self.list_n_traj_ep:
            ax.axhline(y=n_ep, color='red', linestyle='--', linewidth=1)
        ax.set_title("Number of episodes per train interaction")
        ax.set_xlim([0,self.iter_max])
        ax.set_xticks([])

        list_axes[1].set_visible(False)
        
        ax=list_axes[2]
        train_colored_matrix=colored_matrix[0:len(self.list_n_traj_ep)+2]
        im = ax.imshow(train_colored_matrix, aspect='auto')
        row_labels = ['Truth']+[str(n_ep)+' train ep.' for n_ep in self.list_n_traj_ep]+['Learning loss']
        ax.set_yticks(np.arange(train_colored_matrix.shape[0]))  # Establece las posiciones de las filas
        ax.set_yticklabels(row_labels)  # Establece las etiquetas de las filas
        ax.set_xticks([])

        ax=list_axes[3]
        train_matrix=matrix[2:len(self.list_n_traj_ep)+2]
        #train_matrix=train_matrix[::2]
        ax.plot(list(range(self.iter_max)), matrix[1], linewidth=1,color='black',label='Truth')
        train_colors=[Converter.generate_colormap(i, 'Blues_r', -0.2, len(train_matrix)) for i in range(len(train_matrix))]
        for i in range(len(train_matrix)):
            ax.plot(list(range(self.iter_max)), train_matrix[i], linewidth=1,color=train_colors[i],label=str(self.list_n_traj_ep[i])+' train ep.')
        ax.legend(title='Estimates',ncol=2)
        ax.set_title('Evolution of estimates')
        ax.set_xticks([])

        ax=list_axes[4]
        test_colored_matrix=colored_matrix[[0] + list(range(len(self.list_n_traj_ep)+2, colored_matrix.shape[0]))]
        im = ax.imshow(test_colored_matrix, aspect='auto')
        row_labels = ['Truth']+[str(n_ep)+' test ep.' for n_ep in self.list_n_val_ep]
        ax.set_yticks(np.arange(test_colored_matrix.shape[0]))  # Establece las posiciones de las filas
        ax.set_yticklabels(row_labels)  # Establece las etiquetas de las filas
        ax.set_xlabel("Total iterations")

        ax=list_axes[5]
        test_matrix=matrix[-test_colored_matrix.shape[0]+1:]
        #test_matrix=test_matrix[::2]
        test_colors=[Converter.generate_colormap(i, 'Blues_r', -0.2, len(test_matrix)) for i in range(len(test_matrix))]
        ax.plot(list(range(self.iter_max)), matrix[1], linewidth=1,color='black',label='Truth')
        for i in range(len(test_matrix)):
            ax.plot(list(range(self.iter_max)), test_matrix[i], linewidth=1,color=test_colors[i],label=str(self.list_n_val_ep[i])+' test ep.')
        ax.legend(title='Estimates',ncol=2)
        ax.set_xlabel("Total iterations")

        # Crear barra de color para la escala de grises invertida
        Converter.generate_colorbar(fig,[0.1, 0.11, 0.4, 0.01],'gray_r',[0,1],'Normalized estimates',orientation='horizontal') 

        plt.savefig('experiments_intuition/results/CriteriaComparison/figures/'+self.algo+'_'+self.env+'_seed'+str(self.seed)+'/estimation_train_vs_test_analysis.pdf')
        #plt.show()
        plt.close()

    def MAEB_graph_train_vs_test_estimates(self):
        plt.rc('font', family='serif',size=25)
        plt.rc('text', usetex=True)
    
        # Genara la matriz numerica a partir de los datos
        matrix=self.matrix_for_train_vs_test_analysis()
        # Generar matriz de colores
        colored_matrix = self.colored_matrix_for_train_vs_test_analysis(matrix[1:])

        # Dibujar graficas
        fig, axes = plt.subplots(2,1,figsize=(20, 12))
        plt.subplots_adjust(left=0.11, bottom=0.2, right=0.81, top=0.9, wspace=0.39, hspace=0.2)

        list_axes=[ax for _,ax in enumerate(axes.flat)]

        
        ax=list_axes[0]
        train_colored_matrix=colored_matrix[0:len(self.list_n_traj_ep)+1]
        im = ax.imshow(train_colored_matrix, aspect='auto')
        row_labels = ['Truth']+[str(n_ep)+' train ep.' for n_ep in self.list_n_traj_ep]
        ax.set_yticks(np.arange(train_colored_matrix.shape[0]))  # Establece las posiciones de las filas
        ax.set_yticklabels(row_labels)  # Establece las etiquetas de las filas
        ax.set_xlabel("Iteraciones")

        ax=list_axes[1]
        test_colored_matrix=colored_matrix[[0] + list(range(len(self.list_n_traj_ep)+2, colored_matrix.shape[0]))]
        im = ax.imshow(test_colored_matrix, aspect='auto')
        row_labels = ['Truth']+[str(n_ep)+' test ep.' for n_ep in self.list_n_val_ep]
        ax.set_yticks(np.arange(test_colored_matrix.shape[0]))  # Establece las posiciones de las filas
        ax.set_yticklabels(row_labels)  # Establece las etiquetas de las filas
        ax.set_xlabel("Iteraciones")

        # Crear barra de color para la escala de grises invertida
        Converter.generate_colorbar(fig,[0.1, 0.11, 0.4, 0.02],'gray_r',[0,1],'Estimadores normalizados',orientation='horizontal') 

        plt.savefig('experiments_intuition/results/MAEB/'+self.algo+'_'+self.env+'_seed'+str(self.seed)+'_estimation_train_vs_test_analysis.pdf')
        plt.show()
        plt.close()

    def MAEB_graph_cost_analysis(self):
        plt.rc('font', family='serif',size=25)
        plt.rc('text', usetex=True)
        
        # Genara la matriz numerica a partir de los datos
        matrix=self.matrix_for_cost_analysis()

        # Generar matriz de colores
        colored_matrix = self.MAEB_colored_matrix_for_cost_analysis(matrix)

        fig, ax = plt.subplots(figsize=(20, 6))
        plt.subplots_adjust(left=0.11, bottom=0.2, right=0.81, top=0.82, wspace=0.39, hspace=0.2)
        im = ax.imshow(colored_matrix[[0, 2, 5, 6, 7, 8, 9, 10]], aspect='auto')

        # Crear barras de colores
        Converter.generate_colorbar(fig,[0.82, 0.15, 0.015, 0.7],'gray_r',[0,1],'Característica de política normalizada')
        Converter.generate_colorbar(fig,[0.9, 0.15, 0.015, 0.7],'Greens_r',[0,.25],'')
        Converter.generate_colorbar(fig,[0.95, 0.15, 0.015, 0.7],'Reds',[.25,1],'Porcentage de tiempo de iteración (truncado en 1)')
               
        # Etiquetas para los ejes
        row_labels = ['$f$', 'Longitud ep.']+[str(i)+' test ep.' for i in self.list_n_val_ep]
        ax.set_yticks(np.arange(colored_matrix[[0, 2, 5, 6, 7, 8, 9, 10]].shape[0]))  # Establece las posiciones de las filas
        ax.set_yticklabels(row_labels)  # Establece las etiquetas de las filas
        ax.set_xlabel("Política en secuencia")
        
        plt.savefig('experiments_intuition/results/MAEB/'+self.algo+'_'+self.env+'_seed'+str(self.seed)+'_estimation_cost_analysis.pdf')
        plt.close()
   
class CriteriaTuner():
    '''
    Esta clase esta enfocada a fijar la configuracion optima de los criterios train (n_ep) y test (n_ep y freq),
    analizando la evolucion de la magnitud para un grid de posibles valores de cada parametro. El tuning se 
    analiza graficamente, y las mejores configuraciones basadas en ese analisis se almacenan en un csv.
    '''

    def __init__(self,algo,env,seed,resources,
                list_n_val_ep, list_n_val_freq,
                iter_max,perc_time_start=0.1):
    
        # Primero generar los datos necesarios para las graficas
        for n_ep in tqdm(list_n_val_ep):
            Estimator.compute_estimates(algo,env,seed,resources,n_ep,'train')
            Estimator.compute_estimates(algo,env,seed,resources,n_ep,'test')

        # Guardar en una variable las 4 bases de datos
        self.generator=EvolutionGenerator(algo,env,seed,resources,perc_time_start)

        # Guardar el resto de variables
        self.algo=algo
        self.env=env
        self.seed=seed
        self.resources=resources
        self.list_n_val_ep=list_n_val_ep
        self.list_n_val_freq=list_n_val_freq
        self.iter_max=iter_max
        self.start_iter=int(iter_max*perc_time_start)
        self.start_time=self.generator.df_train.loc[self.start_iter-1,'time_seconds']

    #----------------------------------------------------------------------------------------------
    # Calculo de matrices numericas
    #----------------------------------------------------------------------------------------------
    def matrix_magnitude_evolution(self,list_n_ep):
        '''
        Genera 2 matrices numericas (pensado para criterio "best_train"):
         - Una con evoluciones de magnitudes para cada n_ep
         - Otra con los tiempos transcurridos totales totales 
         '''
        matrix_scores=[]
        matrix_times=[]
        x_times=Estimator.time_discretizer(self.algo,self.env,self.seed,self.resources,1,self.iter_max,self.start_time)
        for n_ep in list_n_ep:
            magnitudes=self.generator.magnitude_evolution(x_times,n_ep=n_ep,criteria='best_train',normalized=True)

            # Vector de evolucion de magnitudes
            matrix_scores.append(magnitudes)
            # Vector de tiempos totales transcurridos
            matrix_times.append(x_times)

        return matrix_scores,matrix_times
    
    def matrix_magnitude_valcost_evolution(self,n_ep,list_freq):
        '''
        Genera 2 matrices numericas (pensado para criterio "best_val"):
         - Una con evoluciones de magnitudes y porcentages de coste de validacion, para cada frecuencia
         - Otra con los tiempos transcurridos totales totales 

         A diferencia de las graficas de EstimatorAnalyzer, aqui se tiene en ceunta el tiempo. 
         '''
        matrix_scores=[]
        matrix_times=[]
        x_times=Estimator.time_discretizer(self.algo,self.env,self.seed,self.resources,1,self.iter_max,self.start_time)
        for freq in list_freq:
            x_times_with_freq=Estimator.time_discretizer(self.algo,self.env,self.seed,self.resources,freq,self.iter_max,self.start_time)
            magnitudes,val_times=self.generator.magnitude_evolution(x_times,n_ep,x_times_with_freq,'best_val',True)

            # Vector de evolucion de magnitudes
            matrix_scores.append(magnitudes)
            # Vector de evolucion de porcentage de coste de validacion
            matrix_scores.append([val_times[i]/(x_times[i]+val_times[i]-self.start_time) for i in range(len(x_times))])

            # Vector de tiempos totales transcurridos
            matrix_times.append([i+j for i,j in zip(x_times,val_times)])
            matrix_times.append([i+j for i,j in zip(x_times,val_times)])

        return matrix_scores,matrix_times
    
    #----------------------------------------------------------------------------------------------
    # Representacion grafica de matriz (para uno solo)
    #----------------------------------------------------------------------------------------------
    def plot_magnitude_evolution(self,ax,matrix_scores,matrix_times,min_time,max_time):
        for i in range(len(matrix_scores)):  
            for j in range(len(matrix_scores[i]) - 1):  
                x_start = matrix_times[i][j]
                x_end = matrix_times[i][j + 1]
                gray_value = 1 - matrix_scores[i][j]  # Invertir el valor para la escala de grises
                color = [gray_value, gray_value, gray_value]
                rect = patches.Rectangle((x_start, -(i + 1)), x_end - x_start, 1, facecolor=color, edgecolor=None)
                ax.add_patch(rect)
        ax.set_xlim(min_time, max_time)
        ax.set_ylim(-len(matrix_scores), 0)  # Ajustado para que no haya espacio en blanco
        ax.set_yticks(-np.arange(len(matrix_scores)) - 0.5)  # Centrar los labels en cada fila
        ax.set_yticklabels([str(n_ep) for n_ep in self.list_n_val_ep],fontsize=12)
        ax.set_ylabel( 'Estimations with\ntrain ep.', fontsize=14)

    def plot_magnitude_valcost_evolution(self,ax,matrix_scores,matrix_times,min_time,max_time,n_ep,cost_perc_threshold):
        for i in range(len(matrix_scores)):  
            for j in range(len(matrix_scores[i]) - 1):  
                x_start = matrix_times[i][j]
                x_end = matrix_times[i][j + 1]

                if i%2==0:
                    gray_value = 1 - matrix_scores[i][j]  # Invertir el valor para la escala de grises
                    color = [gray_value, gray_value, gray_value]
                else:
                    value = matrix_scores[i][j]
                    if value <= cost_perc_threshold:
                        # Usar la escala verde invertida (ahora 0 es más oscuro y 1 es más claro)
                        color = Converter.generate_colormap(cost_perc_threshold - value, 'Greens', 0, cost_perc_threshold)[:3]
                    else:
                        if value>1: # Cortar el porcentage en 1
                            value=1
                        color = Converter.generate_colormap(value, 'Reds', cost_perc_threshold, 1)[:3]
                rect = patches.Rectangle((x_start, -(i + 1)), x_end - x_start, 1, facecolor=color, edgecolor=None)
                ax.add_patch(rect)
        ax.set_xlim(min_time, max_time)
        ax.set_ylim(-len(matrix_scores), 0)  # Ajustado para que no haya espacio en blanco
        ax.set_yticks(-np.arange(len(matrix_scores)) - 0.5)  # Centrar los labels en cada fila
        ax.set_yticklabels([str(freq) for freq in self.list_n_val_freq for _ in range(2)],fontsize=12)
        ax.set_ylabel(str(n_ep)+ ' val. ep.;\n Val. freq.:', fontsize=14)

    def MAEB_plot_magnitude_valcost_evolution(self,ax,matrix_scores,matrix_times,min_time,max_time,n_ep,cost_perc_threshold):
        for i in range(len(matrix_scores)):  
            for j in range(len(matrix_scores[i]) - 1):  
                x_start = matrix_times[i][j]
                x_end = matrix_times[i][j + 1]

                if i%2==0:
                    gray_value = 1 - matrix_scores[i][j]  # Invertir el valor para la escala de grises
                    color = [gray_value, gray_value, gray_value]
                    rect = patches.Rectangle((x_start, -(i//2 + 1)), x_end - x_start, 1, facecolor=color, edgecolor=None)
                    ax.add_patch(rect)
        ax.set_xlim(min_time, max_time)
        ax.set_ylim(-len(matrix_scores)/2, 0)  # Ajustado para que no haya espacio en blanco
        ax.set_yticks(-np.arange(len(matrix_scores)/2) - 0.5)  # Centrar los labels en cada fila
        ax.set_yticklabels([str(freq) for freq in self.list_n_val_freq ])
        ax.set_ylabel(str(n_ep)+ ' test ep.\ny frecuencia')

    #----------------------------------------------------------------------------------------------
    # Resumen grafico de una representacion de matriz 
    # (evolucion magnitud->suma de magnitudes (area); evolucion de porcentage de coste de validacion-> media de los porcentajes)
    #----------------------------------------------------------------------------------------------
    def plot_magnitude_valcost_evolution_summary(self,ax,matrix_scores,max_mag_sum,cost_perc_threshold=0.25,highlight_bar=None,best_train=False):
        all_magnitude_sums=[]
        all_porc_means=[]
        for i in range(len(matrix_scores)):
            if i%2==0 or best_train:
                all_magnitude_sums.append(sum(matrix_scores[i]))
            else:
                all_porc_means.append(np.mean(matrix_scores[i]))
        if best_train:
            all_porc_means=[0]*len(all_magnitude_sums)
        all_magnitude_sums=np.array(all_magnitude_sums)[::-1]/max_mag_sum
        all_porc_means=np.array(all_porc_means)[::-1]

        # Dibujar barras 
        for i in range(len(all_magnitude_sums)):
            # Suma magnitudes hacia izquierda
            mag_sum=all_magnitude_sums[i]
            color = [1-mag_sum, 1-mag_sum, 1-mag_sum]
            if highlight_bar==(len(all_magnitude_sums)-1-i):
                ax.barh(np.arange(len(all_magnitude_sums))[i] ,-mag_sum , color=color,edgecolor='blue')
            else:
                ax.barh(np.arange(len(all_magnitude_sums))[i] ,-mag_sum , color=color)

            # Media de porcentage de coste hacia derecha
            porc_mean=all_porc_means[i]
            if porc_mean <= cost_perc_threshold:
                # Usar la escala verde invertida (ahora 0 es más oscuro y 1 es más claro)
                color = Converter.generate_colormap(cost_perc_threshold - porc_mean, 'Greens', 0, cost_perc_threshold)[:3]
            else:
                if porc_mean>1: # Cortar el porcentage en 1
                    porc_mean=1
                color = Converter.generate_colormap(porc_mean, 'Reds', cost_perc_threshold, 1)[:3]

            if highlight_bar==(len(all_magnitude_sums)-1-i):
                ax.barh(np.arange(len(all_magnitude_sums))[i] ,porc_mean , color=color,edgecolor='blue')
                if best_train:
                    ax.text(porc_mean+0.1,np.arange(len(all_magnitude_sums))[i], 'Opt. num. ep.', va='center', fontsize=12, color='blue')
                else:
                    ax.text(porc_mean+0.1,np.arange(len(all_magnitude_sums))[i], 'Trade-off\n(val. ep. vs freq.)', va='center', fontsize=12, color='blue')
            else:
                ax.barh(np.arange(len(all_magnitude_sums))[i] ,porc_mean , color=color)

        ax.axvline(0, color='black', linewidth=1)# Linea vertical separadora
        ax.set_yticks(np.arange(len(all_magnitude_sums)) )
        if best_train:
            ax.set_yticklabels([str(n_ep) for n_ep in self.list_n_val_ep][::-1])
        else:
            ax.set_yticklabels([str(freq) for freq in self.list_n_val_freq][::-1])
        ax.set_xlim([-1,1])
        xticks = ax.get_xticks()
        ax.set_xticklabels([str(abs(tick)) for tick in xticks])# Valores OX positivos en ambos lados

    #----------------------------------------------------------------------------------------------
    # Representacion unificada (se pueden comparar los diferentes criterios, y se destacan las mejores configuraciones)
    #----------------------------------------------------------------------------------------------
    def graph_best_val_tuning(self,cost_perc_threshold=0.25):

        fig, axes = plt.subplots(len(self.list_n_val_ep)+1,2,figsize=(20, 20),gridspec_kw={'width_ratios': [3, 1]})
        plt.subplots_adjust(left=0.076, bottom=0.15, right=0.97, top=0.98, wspace=0.07, hspace=0.12)

        # Calcular totas las matrices de scores y tiempos
        max_times=[]
        min_times=[]
        all_matrix_scores=[]
        all_matrix_times=[]
        for i in range(len(self.list_n_val_ep)):
            matrix_scores,matrix_times=self.matrix_magnitude_valcost_evolution(self.list_n_val_ep[i],self.list_n_val_freq)
            all_matrix_scores.append(matrix_scores)
            all_matrix_times.append([[0]+times for times in matrix_times])
            max_times+=[max(i) for i in matrix_times]

        matrix_scores,matrix_times=self.matrix_magnitude_evolution(self.list_n_val_ep)
        max_times+=[max(i) for i in matrix_times]
        min_times+=[min(i) for i in matrix_times]
        all_matrix_scores.append(matrix_scores)
        all_matrix_times.append([[0]+times for times in matrix_times])

        # Recortar matrices a mismo limite maximo de tiempo
        max_time=min(max_times)
        cut_all_matrix_scores = []
        cut_all_matrix_times = []
        for matrix_scores, matrix_times in zip(all_matrix_scores, all_matrix_times):
            cut_matrix_scores = []
            cut_matrix_times = []
            
            for scores, times in zip(matrix_scores, matrix_times):
                if not np.all(np.array(times)<=max_time):
                    cut_index=list(np.array(times)<=max_time).index(False)
                    cut_matrix_scores.append(scores[:cut_index]+[scores[cut_index-1]]) # El ultimo elemento sumado es para que todas las graficas terminene en el mismo valor horizontal
                    cut_matrix_times.append(times[:cut_index]+[max_time])
                else:
                    cut_matrix_scores.append(scores+[scores[cut_index-1]])
                    cut_matrix_times.append(times+[max_time])

            cut_all_matrix_scores.append(cut_matrix_scores)
            cut_all_matrix_times.append(cut_matrix_times)  

        # Areas de magnitud de todas los criterios test definidos, y la configuracion optima
        mag_sums_test=[]
        for i in range(len(cut_all_matrix_scores)-1):
            matrix_scores=cut_all_matrix_scores[i]
            for row in range(len(matrix_scores)):
                if row%2==0:
                    mag_sums_test.append(sum(matrix_scores[row]))

        indx_tradeoff=mag_sums_test.index(min(mag_sums_test))
        opt_n_ep_test=math.ceil((indx_tradeoff+1)/len(self.list_n_val_freq))
        opt_freq_test= indx_tradeoff-((indx_tradeoff+1)//len(self.list_n_val_freq))*len(self.list_n_val_freq)
        opt_freq_test=[len(self.list_n_val_freq)-1 if opt_freq_test==-1 else opt_freq_test][0]

        # Areas de magnitud de todos los criterios train definidos, y la consiguracion optima
        mag_sums_train=[sum(mag) for mag in cut_all_matrix_scores[-1]]
        opt_n_ep_train=mag_sums_train.index(min(mag_sums_train))

        # Custruir graficas a partir de las matrices
        max_mag_sum=max(mag_sums_train+mag_sums_test)
        list_axes=[ax for _,ax in enumerate(axes.flat)]
        for i in range(int(len(list_axes)/2)):
            matrix_scores=cut_all_matrix_scores[i]
            matrix_times=cut_all_matrix_times[i]

            # De matrices a grafica de cuadrados (de longitudes medidas con tiempos)
            ax=list_axes[2*i]
            if i==int(len(list_axes)/2)-1:
                self.plot_magnitude_evolution(ax,matrix_scores,matrix_times,min(min_times),max_time)
            else:
                self.plot_magnitude_valcost_evolution(ax,matrix_scores,matrix_times,min(min_times),max_time,self.list_n_val_ep[i],cost_perc_threshold)

            if i==0:
                ax.set_title('Evolution of magnitude and validation time percentage')
            if i==int(len(list_axes)/2)-1:
                ax.set_xlabel("Elapsed total time", fontsize=14)
            # De matrices a grafica trade-off
            ax=list_axes[2*i+1]
            if i==opt_n_ep_test-1:
                self.plot_magnitude_valcost_evolution_summary(ax,matrix_scores,max_mag_sum,highlight_bar=opt_freq_test)
            elif i<int(len(list_axes)/2)-1:
                self.plot_magnitude_valcost_evolution_summary(ax,matrix_scores,max_mag_sum)
            else:
                self.plot_magnitude_valcost_evolution_summary(ax,matrix_scores,max_mag_sum,highlight_bar=opt_n_ep_train,best_train=True)
                
            if i==0:
                ax.set_title('Summary of evolutions\nMagnitude vs Validation time percentage')
            if i==int(len(list_axes)/2)-1:
                ax.set_xlabel("Sum of magnitudes         Mean of percentages", fontsize=14)
    
        
        # Crear barras de color 
        Converter.generate_colorbar(fig,[0.1, 0.11, 0.7, 0.01],'gray_r',[0,1],'Normalized magnitudes',orientation='horizontal') 
        Converter.generate_colorbar(fig,[0.1, 0.075, 0.7, 0.01],'Greens_r',[0,.25],'',orientation='horizontal') 
        Converter.generate_colorbar(fig,[0.1, 0.05, 0.7, 0.01],'Reds',[.25,1],'Percentage of validation time (clipped in 1)',orientation='horizontal') 
        
        plt.savefig('experiments_intuition/results/CriteriaComparison/figures/'+self.algo+'_'+self.env+'_seed'+str(self.seed)+'/criteria_tuner.pdf')
        #plt.show()
        plt.close()

        # Guardar configuraciones optimas en la csv si existe, si no crearla.
        opt_conf=[self.list_n_val_ep[opt_n_ep_train],self.list_n_val_ep[opt_n_ep_test-1],self.list_n_val_freq[opt_freq_test]]
        path_csv='experiments_intuition/results/CriteriaComparison/data/criteria_conf_by_process.csv'
        if not os.path.exists(path_csv):
            df = pd.DataFrame(columns=['process_id','train_n_ep','test_n_ep','test_freq'])
            df.to_csv(path_csv, index=False)
        
        with open(path_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([self.algo+'_'+self.env+'_seed'+str(self.seed)]+opt_conf) 
        Converter.from_csv_to_png('experiments_intuition/results/CriteriaComparison/data','criteria_conf_by_process')
        return opt_conf
    
    def MAEB_graph_best_val_tuning(self,cost_perc_threshold=0.25):

        plt.rc('font', family='serif',size=25)
        plt.rc('text', usetex=True)

        fig, axes = plt.subplots(len(self.list_n_val_ep)+1,2,figsize=(20, 10),gridspec_kw={'width_ratios': [3, 1]})
        plt.subplots_adjust(left=0.076, bottom=0.15, right=0.97, top=0.98, wspace=0.07, hspace=0.2)

        # Calcular totas las matrices de scores y tiempos
        max_times=[]
        min_times=[]
        all_matrix_scores=[]
        all_matrix_times=[]
        for i in range(len(self.list_n_val_ep)):
            matrix_scores,matrix_times=self.matrix_magnitude_valcost_evolution(self.list_n_val_ep[i],self.list_n_val_freq)
            all_matrix_scores.append(matrix_scores)
            all_matrix_times.append([[0]+times for times in matrix_times])
            max_times+=[max(i) for i in matrix_times]

        matrix_scores,matrix_times=self.matrix_magnitude_evolution(self.list_n_val_ep)
        max_times+=[max(i) for i in matrix_times]
        min_times+=[min(i) for i in matrix_times]
        all_matrix_scores.append(matrix_scores)
        all_matrix_times.append([[0]+times for times in matrix_times])

        # Recortar matrices a mismo limite maximo de tiempo
        max_time=min(max_times)
        cut_all_matrix_scores = []
        cut_all_matrix_times = []
        for matrix_scores, matrix_times in zip(all_matrix_scores, all_matrix_times):
            cut_matrix_scores = []
            cut_matrix_times = []
            
            for scores, times in zip(matrix_scores, matrix_times):
                if not np.all(np.array(times)<=max_time):
                    cut_index=list(np.array(times)<=max_time).index(False)
                    cut_matrix_scores.append(scores[:cut_index]+[scores[cut_index-1]]) # El ultimo elemento sumado es para que todas las graficas terminene en el mismo valor horizontal
                    cut_matrix_times.append(times[:cut_index]+[max_time])
                else:
                    cut_matrix_scores.append(scores+[scores[cut_index-1]])
                    cut_matrix_times.append(times+[max_time])

            cut_all_matrix_scores.append(cut_matrix_scores)
            cut_all_matrix_times.append(cut_matrix_times)  

        # Areas de magnitud de todas los criterios test definidos, y la configuracion optima
        mag_sums_test=[]
        for i in range(len(cut_all_matrix_scores)-1):
            matrix_scores=cut_all_matrix_scores[i]
            for row in range(len(matrix_scores)):
                if row%2==0:
                    mag_sums_test.append(sum(matrix_scores[row]))

        indx_tradeoff=mag_sums_test.index(min(mag_sums_test))
        opt_n_ep_test=math.ceil((indx_tradeoff+1)/len(self.list_n_val_freq))
        opt_freq_test= indx_tradeoff-((indx_tradeoff+1)//len(self.list_n_val_freq))*len(self.list_n_val_freq)
        opt_freq_test=[len(self.list_n_val_freq)-1 if opt_freq_test==-1 else opt_freq_test][0]

        # Areas de magnitud de todos los criterios train definidos, y la consiguracion optima
        mag_sums_train=[sum(mag) for mag in cut_all_matrix_scores[-1]]
        opt_n_ep_train=mag_sums_train.index(min(mag_sums_train))

        # Custruir graficas a partir de las matrices
        max_mag_sum=max(mag_sums_train+mag_sums_test)
        list_axes=[ax for _,ax in enumerate(axes.flat)]
        for i in range(int(len(list_axes)/2)):
            matrix_scores=cut_all_matrix_scores[i]
            matrix_times=cut_all_matrix_times[i]

            # De matrices a grafica de cuadrados (de longitudes medidas con tiempos)
            ax=list_axes[2*i]

            if i==int(len(list_axes)/2)-1:
                ax.axis('off')
            else:
                self.MAEB_plot_magnitude_valcost_evolution(ax,matrix_scores,matrix_times,min(min_times),max_time,self.list_n_val_ep[i],cost_perc_threshold)

            list_axes[2*i+1].axis("off")
    
        
        # Crear barras de color 
        plt.savefig('experiments_intuition/results/MAEB/'+self.algo+'_'+self.env+'_seed'+str(self.seed)+'_criteria_tuner.pdf')
        plt.close()

    
class ProcessIndependentAnalyzer():
    def __init__(self,iter_max,perc_time_start=0.1,
                 global_deg_metric='mean_update_deg',local_deg_metric='greater_prob',
                 all_possible_conf=False, # Para generar datos relacionados con el analisis de sensibilidad
                 grid_train_n_ep=None,grid_test_n_ep=None,grid_test_freq=None # Para almacenar datos relacionados con el analisis para ganar intuicion
                 ):

        start_iter=int(iter_max*perc_time_start)-1
        self.iter_max=iter_max
        self.start_iter=start_iter
        
        # Mirar si ya se ha llamado a esta funcion previamente (si existen las bases de datos necesrias, si no inicializarlas)
        df_degradation=Estimator.read_create_estimates_csv('experiments_intuition/results/CriteriaComparison/data/deg_mag/level_degradation.csv',iter_max,start_iter)
        df_last_mag=Estimator.read_create_estimates_csv('experiments_intuition/results/CriteriaComparison/data/deg_mag/criteria_last_mag.csv',iter_max,start_iter)
        df_train_mag=Estimator.read_create_estimates_csv('experiments_intuition/results/CriteriaComparison/data/deg_mag/criteria_best_train_mag.csv',iter_max,start_iter)
        df_test_mag=Estimator.read_create_estimates_csv('experiments_intuition/results/CriteriaComparison/data/deg_mag/criteria_best_test_mag.csv',iter_max,start_iter)
        
        df_last_eff=Estimator.read_create_estimates_csv('experiments_intuition/results/CriteriaComparison/data/deg_mag/criteria_last_eff.csv',iter_max,start_iter)
        df_train_eff=Estimator.read_create_estimates_csv('experiments_intuition/results/CriteriaComparison/data/deg_mag/criteria_best_train_eff.csv',iter_max,start_iter)
        df_test_eff=Estimator.read_create_estimates_csv('experiments_intuition/results/CriteriaComparison/data/deg_mag/criteria_best_test_eff.csv',iter_max,start_iter)
        
        df_test_start_end=Converter.generate_df('experiments_intuition/results/CriteriaComparison/data/test_start_end_perc_time.csv',
                                                ['process_id','test_conf','start_test','end_test','perc_good','invest_time','last_good','last_time'])

        # Leer base de datos donde se almacenan las mejores configuraciones de los criterios por proceso (criteria_conf_by_process.csv)
        df_conf=pd.read_csv('experiments_intuition/results/CriteriaComparison/data/criteria_conf_by_process.csv')
        '''
        
        # Calcular y guardar por proceso: evolucion de nivel de degradacion y evolucion de magnitudes por criterio (si no estan ya guardados)
        for row in tqdm(range(df_conf.shape[0])):# Aqui estan guardados todos los procesos que consideraremos
            process_id=df_conf.loc[row,'process_id']
            print(process_id)
            algo,env,seed=Converter.process_id_splitter(process_id)

            generator=EvolutionGenerator(algo,env,seed,'16cpu1gpu_mejorado',perc_time_start)
            x_times=generator.df_train['time_seconds'].tolist()[start_iter:iter_max]  
            min_time=generator.df_train.loc[start_iter,'time_seconds']
            
            # Si los datos del proceso no estan ya almacenados calcularlos y almacenarlos (configuraciones optimas en cada proceso)
            if  process_id not in list(df_degradation.columns):
                
                # Calcular y guardar las evolucion de nivel de degradacion
                df_degradation[process_id+'_'+global_deg_metric+'_'+local_deg_metric]=generator.degradation_evolution(x_times,global_deg_metric,local_deg_metric)
            
                # Configuracion optima de los criterios train y test 
                train_n_ep,test_n_ep,test_freq=df_conf.loc[row,['train_n_ep','test_n_ep','test_freq']]

                # Calcular y guardar la evolucion de magnitud de cada criterio (las estimaciones para las configuraciones anteriores las tengo guardadas porque ya he tenido que ejecutar el tuner)
                x_times_with_freq=Estimator.time_discretizer(algo,env,seed,'16cpu1gpu_mejorado',test_freq,iter_max,min_time)
                df_last_mag[process_id]=generator.magnitude_evolution(x_times,criteria='last',normalized=True)
                df_train_mag[process_id]=generator.magnitude_evolution(x_times,n_ep=train_n_ep,criteria='best_train',normalized=True)
                df_test_mag[process_id]=generator.magnitude_evolution(x_times,n_ep=test_n_ep,freq=x_times_with_freq,criteria='best_val',normalized=True)[0]

                df_last_eff[process_id]=generator.effectiveness_evolution(x_times,criteria='last',normalized=True)
                df_train_eff[process_id]=generator.effectiveness_evolution(x_times,n_ep=train_n_ep,criteria='best_train',normalized=True)
                df_test_eff[process_id]=generator.effectiveness_evolution(x_times,n_ep=test_n_ep,freq=x_times_with_freq,criteria='best_val',normalized=True)[0]
            print('if de avanzado listo')
            
            # Almacenar datos adicionales para posibles combinaciones de configuraciones optimas (para el analisis de sensibilidad)
            if all_possible_conf:
                all_train_n_ep=list(set(df_conf['train_n_ep']))
                all_test_n_ep_freq=list(set(tuple(pair) for pair in zip(df_conf['test_n_ep'],df_conf['test_freq'])))

                # Calcular y guardar la evolucion de magnitud de cada criterio (las estimaciones para las configuraciones anteriores las tengo guardadas porque ya he tenido que ejecutar el tuner)
                for n_ep in all_train_n_ep:
                    if process_id+'_'+str(n_ep) not in list(df_train_mag.columns):
                        df_train_mag[process_id+'_'+str(n_ep)]=generator.magnitude_evolution(x_times,n_ep=n_ep,criteria='best_train',normalized=True)
                    if process_id+'_'+str(n_ep) not in list(df_train_eff.columns):
                        df_train_eff[process_id+'_'+str(n_ep)]=generator.effectiveness_evolution(x_times,n_ep=n_ep,criteria='best_train',normalized=True)

                for n_ep,freq in all_test_n_ep_freq:
                    if process_id+'_'+str(n_ep)+'_'+str(freq) not in list(df_test_mag.columns):
                        x_times_with_freq=Estimator.time_discretizer(algo,env,seed,'16cpu1gpu_mejorado',freq,iter_max,min_time)
                        df_test_mag[process_id+'_'+str(n_ep)+'_'+str(freq)]=generator.magnitude_evolution(x_times,n_ep=n_ep,freq=x_times_with_freq,criteria='best_val',normalized=True)[0]
                    if process_id+'_'+str(n_ep)+'_'+str(freq) not in list(df_test_eff.columns):
                        x_times_with_freq=Estimator.time_discretizer(algo,env,seed,'16cpu1gpu_mejorado',freq,iter_max,min_time)
                        df_test_eff[process_id+'_'+str(n_ep)+'_'+str(freq)]=generator.effectiveness_evolution(x_times,n_ep=n_ep,freq=x_times_with_freq,criteria='best_val',normalized=True)[0]


            print('if de sensibilidad listo')
            
            # Almacenar datos de configuraciones indicadas (para analisis intermedio, menos avanzado)
            if grid_train_n_ep!=None:
                for n_ep in grid_train_n_ep:
                    if process_id+'_'+str(n_ep) not in list(df_train_mag.columns):
                        df_train_mag[process_id+'_'+str(n_ep)]=generator.magnitude_evolution(x_times,n_ep=n_ep,criteria='best_train',normalized=True)
                    if process_id+'_'+str(n_ep) not in list(df_train_eff.columns):
                        df_train_eff[process_id+'_'+str(n_ep)]=generator.effectiveness_evolution(x_times,n_ep=n_ep,criteria='best_train',normalized=True)
            print('if de intuicion train listo')  
            if grid_test_n_ep!=None:
                for n_ep in grid_test_n_ep:

                    # Sin contar el extra de tiempo de validacion con freq=1
                    if process_id+'_'+str(n_ep)+'_without_extra' not in list(df_test_mag.columns):
                        x_times_with_freq=Estimator.time_discretizer(algo,env,seed,'16cpu1gpu_mejorado',1,iter_max,min_time)
                        df_test_mag[process_id+'_'+str(n_ep)+'_without_extra']=generator.magnitude_evolution(x_times,n_ep=n_ep,freq=x_times_with_freq,criteria='best_val',normalized=True)[0]
                    if process_id+'_'+str(n_ep)+'_without_extra' not in list(df_test_eff.columns):  
                        x_times_with_freq=Estimator.time_discretizer(algo,env,seed,'16cpu1gpu_mejorado',1,iter_max,min_time)
                        df_test_eff[process_id+'_'+str(n_ep)+'_without_extra']=generator.effectiveness_evolution(x_times,n_ep=n_ep,freq=x_times_with_freq,criteria='best_val',normalized=True)[0]

            
                    # Contando el extra de tiempo de validacion para diferentes frecuencias
                    for freq in grid_test_freq:
                        if process_id+'_'+str(n_ep)+'_'+str(freq) not in list(df_test_mag.columns):
                            x_times_with_freq=Estimator.time_discretizer(algo,env,seed,'16cpu1gpu_mejorado',freq,iter_max,min_time)
                            df_test_mag[process_id+'_'+str(n_ep)+'_'+str(freq)]=generator.magnitude_evolution(x_times,n_ep=n_ep,freq=x_times_with_freq,criteria='best_val',normalized=True)[0]
                        if process_id+'_'+str(n_ep)+'_'+str(freq) not in list(df_test_eff.columns):
                            x_times_with_freq=Estimator.time_discretizer(algo,env,seed,'16cpu1gpu_mejorado',freq,iter_max,min_time)    
                            df_test_mag[process_id+'_'+str(n_ep)+'_'+str(freq)]=generator.effectiveness_evolution(x_times,n_ep=n_ep,freq=x_times_with_freq,criteria='best_val',normalized=True)[0]

            
            print('if de intuicion test listo')  
            
            # Guardar cambios en bases de datos
            df_degradation.to_csv('experiments_intuition/results/CriteriaComparison/data/deg_mag/level_degradation.csv', index=False)
            df_last_mag.to_csv('experiments_intuition/results/CriteriaComparison/data/deg_mag/criteria_last_mag.csv', index=False)
            df_train_mag.to_csv('experiments_intuition/results/CriteriaComparison/data/deg_mag/criteria_best_train_mag.csv', index=False)
            df_test_mag.to_csv('experiments_intuition/results/CriteriaComparison/data/deg_mag/criteria_best_test_mag.csv', index=False)

            df_last_eff.to_csv('experiments_intuition/results/CriteriaComparison/data/deg_mag/criteria_last_eff.csv', index=False)
            df_train_eff.to_csv('experiments_intuition/results/CriteriaComparison/data/deg_mag/criteria_best_train_eff.csv', index=False)
            df_test_eff.to_csv('experiments_intuition/results/CriteriaComparison/data/deg_mag/criteria_best_test_eff.csv', index=False)

            df_test_start_end.to_csv('experiments_intuition/results/CriteriaComparison/data/test_start_end_perc_time.csv', index=False)
        '''
        self.df_degradation=df_degradation
        self.df_last_mag=df_last_mag
        self.df_train_mag=df_train_mag
        self.df_test_mag=df_test_mag
        self.df_last_eff=df_last_eff
        self.df_train_eff=df_train_eff
        self.df_test_eff=df_test_eff
        self.df_test_start_end=df_test_start_end
        self.iter_max=iter_max

    def generate_criteria_rank_by_time_data(self,process_ids,train_grid_n_ep,test_grid_n_ep,test_grid_freq,global_deg_metric,local_deg_metric):
        
        train_suffix=[str(n_ep) for n_ep in train_grid_n_ep]
        test_suffix=[str(n_ep)+'_'+str(freq) for n_ep in test_grid_n_ep for freq in test_grid_freq]
        
        df = pd.DataFrame(columns=['process_id','seq_size','degradation_level','rank_last']+['rank_'+i for i in train_suffix+test_suffix]+['not_first_prob','first_mag'])
        
        for process_id in process_ids:
            
            # Niveles de degradacion y tamaños desecuencias
            degradation_levels=self.df_degradation[process_id+'_'+global_deg_metric+'_'+local_deg_metric]
            seq_sizes=list(range(1,len(degradation_levels)+1))

            # Magnitudes
            last_criterion_mag=[self.df_last_mag[process_id].tolist()]
            train_criteria_mag=[self.df_train_mag[process_id+'_'+suffix] for suffix in train_suffix]
            test_criteria_mag=[self.df_test_mag[process_id+'_'+suffix] for suffix in test_suffix]
            criteria_mag=np.array(last_criterion_mag+train_criteria_mag+test_criteria_mag).T

            # Rankings y similitudes de magnitudes
            criteria_rankings=[Converter.from_list_to_ranking(magnitudes) for magnitudes in criteria_mag]
            not_first_probs=[Converter.from_list_to_prob_not_first(magnitudes) for magnitudes in criteria_mag]
            first_mags=[max(magnitudes) for magnitudes in criteria_mag]

            # Completar base de datos con datos de proceso
            rows_to_add=[[process_id,seq_size,degradation_level]+criteria_ranking+[not_first_prob,first_mag]
                         for seq_size,degradation_level, criteria_ranking, not_first_prob,first_mag in zip(seq_sizes,degradation_levels,criteria_rankings,not_first_probs,first_mags)]
            df_new = pd.DataFrame(rows_to_add, columns=df.columns)
            df = pd.concat([df, df_new], ignore_index=True)

        return df
    
    def MAEB_generate_criteria_rank_by_time_data(self,process_ids,train_grid_n_ep,test_grid_n_ep,test_grid_freq,global_deg_metric,local_deg_metric):
        
        train_suffix=[str(n_ep) for n_ep in train_grid_n_ep]
        test_suffix=[str(n_ep)+'_'+str(freq) for n_ep in test_grid_n_ep for freq in test_grid_freq]
        
        df = pd.DataFrame(columns=['process_id','seq_size','degradation_level','rank_last']+['rank_'+i for i in train_suffix+test_suffix]+['not_first_prob','first_mag','first_eff'])
        
        for process_id in process_ids:
            
            # Niveles de degradacion y tamaños desecuencias
            degradation_levels=self.df_degradation[process_id+'_'+global_deg_metric+'_'+local_deg_metric]
            seq_sizes=list(range(1,len(degradation_levels)+1))

            # Magnitudes
            last_criterion_mag=[self.df_last_mag[process_id].tolist()]
            train_criteria_mag=[self.df_train_mag[process_id+'_'+suffix] for suffix in train_suffix]
            test_criteria_mag=[self.df_test_mag[process_id+'_'+str(n_ep)+'_without_extra'] for n_ep in test_grid_n_ep]
            criteria_mag=np.array(last_criterion_mag+train_criteria_mag+test_criteria_mag).T

            # Effectiveness
            last_criterion_eff=[self.df_last_eff[process_id].tolist()]
            train_criteria_eff=[self.df_train_eff[process_id+'_'+suffix] for suffix in train_suffix]
            test_criteria_eff=[self.df_test_eff[process_id+'_'+str(n_ep)+'_without_extra'] for n_ep in test_grid_n_ep]
            criteria_eff=np.array(last_criterion_eff+train_criteria_eff+test_criteria_eff).T


            # Rankings y similitudes de magnitudes
            criteria_rankings=[Converter.from_list_to_ranking(magnitudes) for magnitudes in criteria_mag]
            not_first_probs=[Converter.from_list_to_diff_not_best(magnitudes) for magnitudes in criteria_mag]
            first_mags=[max(magnitudes) for magnitudes in criteria_mag]
            first_effs=[eff[list(mag).index(min(mag))] for mag,eff in zip(criteria_mag,criteria_eff)]

            # Completar base de datos con datos de proceso
            rows_to_add=[[process_id,seq_size,degradation_level]+criteria_ranking+[not_first_prob,first_mag,first_eff]
                         for seq_size,degradation_level, criteria_ranking, not_first_prob,first_mag,first_eff in zip(seq_sizes,degradation_levels,criteria_rankings,not_first_probs,first_mags,first_effs)]
            df_new = pd.DataFrame(rows_to_add, columns=df.columns)
            df = pd.concat([df, df_new], ignore_index=True)

        return df

    def MAEB2_generate_criteria_rank_by_time_data(self,process_ids,train_grid_n_ep,test_grid_n_ep,test_grid_freq,global_deg_metric,local_deg_metric):
        
        train_suffix=[str(n_ep) for n_ep in train_grid_n_ep]
        test_suffix=[str(n_ep)+'_'+str(freq) for n_ep in test_grid_n_ep for freq in test_grid_freq]
        
        df = pd.DataFrame(columns=['process_id','seq_size','degradation_level','mag_last']+['mag_'+i for i in train_suffix+test_suffix]+['eff_last']+['eff_'+i for i in train_suffix+test_suffix])
        
        for process_id in process_ids:
            
            # Niveles de degradacion y tamaños de secuencias
            degradation_levels=self.df_degradation[process_id+'_'+global_deg_metric+'_'+local_deg_metric]
            seq_sizes=list(range(1,len(degradation_levels)+1))

            # Magnitudes
            last_criterion_mag=[self.df_last_mag[process_id].tolist()]
            train_criteria_mag=[self.df_train_mag[process_id+'_'+suffix] for suffix in train_suffix]
            test_criteria_mag=[self.df_test_mag[process_id+'_'+str(n_ep)+'_without_extra'] for n_ep in test_grid_n_ep]
            criteria_mag=np.array(last_criterion_mag+train_criteria_mag+test_criteria_mag).T

            # Eficacias
            last_criterion_eff=[self.df_last_eff[process_id].tolist()]
            train_criteria_eff=[self.df_train_eff[process_id+'_'+suffix] for suffix in train_suffix]
            test_criteria_eff=[self.df_test_eff[process_id+'_'+str(n_ep)+'_without_extra'] for n_ep in test_grid_n_ep]
            criteria_eff=np.array(last_criterion_eff+train_criteria_eff+test_criteria_eff).T

            # Completar base de datos con datos de proceso
            rows_to_add=[[process_id,seq_size,degradation_level]+list(c_mag)+list(c_eff)
                         for seq_size,degradation_level,c_mag,c_eff in zip(seq_sizes,degradation_levels,criteria_mag,criteria_eff)]
            df_new = pd.DataFrame(rows_to_add, columns=df.columns)
            df = pd.concat([df, df_new], ignore_index=True)

        return df

    def generate_criteria_rank_by_degradation_data(self,process_ids,global_deg_metric,local_deg_metric):
        # Generar datos necesarios para la grafica
        df = pd.DataFrame(columns=['process_id','degradation_level','rank_last','rank_best_train','rank_best_test','not_first_prob'])

        for process_id in process_ids:
            degradation_levels=self.df_degradation[process_id+'_'+global_deg_metric+'_'+local_deg_metric]

            criteria_mag=np.array([self.df_last_mag[process_id],self.df_train_mag[process_id],self.df_test_mag[process_id]]).T
            criteria_rankings=[Converter.from_list_to_ranking(magnitudes) for magnitudes in criteria_mag]
            not_first_probs=[Converter.from_list_to_prob_not_first(magnitudes) for magnitudes in criteria_mag]

            rows_to_add=[[process_id,degradation_level,criteria_ranking[0],criteria_ranking[1],criteria_ranking[2],not_first_prob]
                         for degradation_level, criteria_ranking, not_first_prob in zip(degradation_levels,criteria_rankings,not_first_probs)]
            df_new = pd.DataFrame(rows_to_add, columns=df.columns)
            df = pd.concat([df, df_new], ignore_index=True)

        return df
    
    def generate_train_test_criteria_rank_by_degradation_data(self,process_ids,global_deg_metric,local_deg_metric,train_or_test):
        
        df_conf=pd.read_csv('experiments_intuition/results/CriteriaComparison/data/criteria_conf_by_process.csv')
        all_train_n_ep=list(set(df_conf['train_n_ep']))
        all_test_n_ep_freq=list(set(tuple(pair) for pair in zip(df_conf['test_n_ep'],df_conf['test_freq'])))
        
        if train_or_test=='train':
            process_id_suffix=[str(n_ep) for n_ep in all_train_n_ep]
        if train_or_test=='test':
            process_id_suffix=[str(n_ep)+'_'+str(freq) for n_ep,freq in all_test_n_ep_freq]
            
        # Generar datos necesarios para la grafica
        df = pd.DataFrame(columns=['process_id','degradation_level']+['rank_'+i for i in process_id_suffix]+['not_first_prob'])

        for process_id in process_ids:
            degradation_levels=self.df_degradation[process_id+'_'+global_deg_metric+'_'+local_deg_metric]

            if train_or_test=='train':
                criteria_mag=np.array([self.df_train_mag[process_id+'_'+suffix] for suffix in process_id_suffix]).T
            if train_or_test=='test':
                criteria_mag=np.array([self.df_test_mag[process_id+'_'+suffix] for suffix in process_id_suffix]).T

            criteria_rankings=[Converter.from_list_to_ranking(magnitudes) for magnitudes in criteria_mag]
            not_first_probs=[Converter.from_list_to_prob_not_first(magnitudes) for magnitudes in criteria_mag]

            rows_to_add=[[process_id,degradation_level]+criteria_ranking+[not_first_prob]
                         for degradation_level, criteria_ranking, not_first_prob in zip(degradation_levels,criteria_rankings,not_first_probs)]
            df_new = pd.DataFrame(rows_to_add, columns=df.columns)
            df = pd.concat([df, df_new], ignore_index=True)

        return df

    def generate_train_test_grid_rank_by_degradation_data(self,process_ids,global_deg_metric,local_deg_metric,train_or_test,grid_n_ep,freq=None):

        if train_or_test=='train':
            process_id_suffix=[str(n_ep) for n_ep in grid_n_ep]
        if train_or_test=='test':
            if freq==None:
                process_id_suffix=[str(n_ep)+'_without_extra' for n_ep in grid_n_ep]
            else:
                process_id_suffix=[str(n_ep)+'_'+str(freq) for n_ep in grid_n_ep]
             
        # Generar datos necesarios para la grafica si ya no estan generados
        df = pd.DataFrame(columns=['process_id','seq_size','degradation_level']+['rank_'+i for i in process_id_suffix]+['not_first_prob'])

        for process_id in process_ids:
            
            degradation_levels=self.df_degradation[process_id+'_'+global_deg_metric+'_'+local_deg_metric]
            seq_sizes=list(range(1,len(degradation_levels)+1))

            if train_or_test=='train':
                criteria_mag=np.array([self.df_train_mag[process_id+'_'+suffix] for suffix in process_id_suffix]).T
            if train_or_test=='test':
                criteria_mag=np.array([self.df_test_mag[process_id+'_'+suffix] for suffix in process_id_suffix]).T

            criteria_rankings=[Converter.from_list_to_ranking(magnitudes) for magnitudes in criteria_mag]
            not_first_probs=[Converter.from_list_to_prob_not_first(magnitudes) for magnitudes in criteria_mag]

            rows_to_add=[[process_id,seq_size,degradation_level]+criteria_ranking+[not_first_prob]
                         for seq_size,degradation_level, criteria_ranking, not_first_prob in zip(seq_sizes,degradation_levels,criteria_rankings,not_first_probs)]
            df_new = pd.DataFrame(rows_to_add, columns=df.columns)
            df = pd.concat([df, df_new], ignore_index=True)

        return df


    def matrix_best_criteria_by_time(self,process_ids,train_grid_n_ep,test_grid_n_ep,test_grid_freq,global_deg_metric,local_deg_metric):
        df=self.generate_criteria_rank_by_time_data(process_ids,train_grid_n_ep,test_grid_n_ep,test_grid_freq,global_deg_metric,local_deg_metric)
        df=df[df['process_id'].isin(process_ids)]

        seq_size_intervals=list(np.arange(0,1.1,0.1))
        max_size=self.iter_max-self.start_iter
        matrix_perc=[] # Aqui se almacenaran los datos para las barras apiladas
        matrix_prob=[] # Aqui se almacenaran los datos para los valores dentro de las barras apiladas
        matrix_mag=[]
        matrix_deg=[]

        deg_intervals=list(np.arange(0,1.2,0.2))[::-1]
        matrix_deg_intervals=[] # Aqui se almacenaran "las distribuciones" de los niveles de degradacion por intervalo de tamaños de secuencia

        split_seq_size=int(max_size*seq_size_intervals[1])
        for i in range(1,len(seq_size_intervals)):
            max_seq_size=split_seq_size*i
            min_seq_size=1+split_seq_size*(i-1)
            df_seq_size=df[(df['seq_size']>=min_seq_size) & (df['seq_size']<max_seq_size) ]

            perc_list=[]
            not_first_prob_list=[]
            first_mag_list=[]
            first_deg_list=[]
            for j in range(3,df_seq_size.shape[1]-2):
                perc_list.append((df_seq_size.iloc[:, j]==1).sum()/df_seq_size.shape[0])
                not_first_prob_list.append(df_seq_size[df_seq_size.iloc[:, j]==1]['not_first_prob'].mean())
                first_mag_list.append(df_seq_size[df_seq_size.iloc[:, j]==1]['first_mag'].mean())
                first_deg_list.append(df_seq_size[df_seq_size.iloc[:, j]==1]['degradation_level'].mean())

            matrix_perc.append(perc_list)
            matrix_prob.append(not_first_prob_list)
            matrix_mag.append(first_mag_list)
            matrix_deg.append(first_deg_list)
            matrix_deg_intervals.append([ sum(deg_intervals[i] <= deg < deg_intervals[i-1] for deg in df_seq_size['degradation_level'] )  for i in range(1,len(deg_intervals))])


        return np.array(matrix_perc),np.array(matrix_prob), np.array(matrix_deg),np.array(matrix_deg_intervals),np.array(matrix_mag),seq_size_intervals, deg_intervals, ['['+str(round(deg_intervals[i],1))+','+str(round(deg_intervals[i-1],1))+')' for i in range(1,len(deg_intervals))], df.columns[3:df_seq_size.shape[1]-2].str.replace('rank_', '', regex=False).tolist()

    def matrix_best_criteria_by_degradation(self,process_ids,global_deg_metric,local_deg_metric):

        df=self.generate_criteria_rank_by_degradation_data(process_ids,global_deg_metric,local_deg_metric)
        df=df[df['process_id'].isin(process_ids)]

        degradation_intervals=list(np.arange(0,1.02,0.02))
        matrix_perc=[]
        matrix_prob=[]
        num_data_per_level=[]
        for i in range(1,len(degradation_intervals)):
            max_deg_level=degradation_intervals[i]
            min_deg_level=degradation_intervals[i-1]
            df_deg_level=df[(df['degradation_level']>=min_deg_level) & (df['degradation_level']<max_deg_level)]

            perc_last_first=(df_deg_level['rank_last']==1).sum()/df_deg_level.shape[0]
            perc_best_train_first=(df_deg_level['rank_best_train']==1).sum()/df_deg_level.shape[0]
            perc_best_test_first=(df_deg_level['rank_best_test']==1).sum()/df_deg_level.shape[0]

            not_first_prob_last=df_deg_level[df_deg_level['rank_last']==1]['not_first_prob'].mean()
            not_first_prob_best_train=df_deg_level[df_deg_level['rank_best_train']==1]['not_first_prob'].mean()
            not_first_prob_best_test=df_deg_level[df_deg_level['rank_best_test']==1]['not_first_prob'].mean()

            matrix_perc.append([perc_last_first,perc_best_train_first,perc_best_test_first])
            matrix_prob.append([not_first_prob_last,not_first_prob_best_train,not_first_prob_best_test])
            num_data_per_level.append(df_deg_level.shape[0])

        return np.array(matrix_perc),np.array(matrix_prob), degradation_intervals, num_data_per_level
    
    def matrix_train_test_criteria_by_degradation(self,process_ids,global_deg_metric,local_deg_metric,train_or_test):
        
        df=self.generate_train_test_criteria_rank_by_degradation_data(process_ids,global_deg_metric,local_deg_metric,train_or_test)
        df=df[df['process_id'].isin(process_ids)]

        degradation_intervals=list(np.arange(0,1.02,0.02))
        matrix_perc=[]
        matrix_prob=[]
        num_data_per_level=[]
        for i in range(1,len(degradation_intervals)):
            max_deg_level=degradation_intervals[i]
            min_deg_level=degradation_intervals[i-1]
            df_deg_level=df[(df['degradation_level']>=min_deg_level) & (df['degradation_level']<max_deg_level)]

            perc_list=[]
            not_first_prob_list=[]
            for j in range(2,df_deg_level.shape[1]-1):
                perc_list.append((df_deg_level.iloc[:, j]==1).sum()/df_deg_level.shape[0])
                not_first_prob_list.append(df_deg_level[df_deg_level.iloc[:, j]==1]['not_first_prob'].mean())

            matrix_perc.append(perc_list)
            matrix_prob.append(not_first_prob_list)
            num_data_per_level.append(df_deg_level.shape[0])

        return np.array(matrix_perc),np.array(matrix_prob), degradation_intervals, num_data_per_level, df.columns[2:df_deg_level.shape[1]-1].str.replace('rank_', '', regex=False).tolist()

    def matrix_train_test_grid_by_degradation(self,process_ids,global_deg_metric,local_deg_metric,seq_size,train_or_test,grid_n_ep,freq=None):

        df=self.generate_train_test_grid_rank_by_degradation_data(process_ids,global_deg_metric,local_deg_metric,train_or_test,grid_n_ep,freq)
        df=df[df['process_id'].isin(process_ids)]

        degradation_intervals=list(np.arange(0,1.05,0.05))
        matrix_perc=[]
        matrix_prob=[]
        num_data_per_level=[]
        for i in range(1,len(degradation_intervals)):
            max_deg_level=degradation_intervals[i]
            min_deg_level=degradation_intervals[i-1]
            df_deg_level=df[(df['degradation_level']>=min_deg_level) & (df['degradation_level']<max_deg_level) & (df['seq_size']<=seq_size)]# TODO: si cogemos exactamente las secuencias de ese tamaño, estamos construyendo las graficas solo con tantas secuencias como procesos consideremos. Si en cambio usamos <= seria acomulativo

            perc_list=[]
            not_first_prob_list=[]
            for j in range(3,df_deg_level.shape[1]-1):
                perc_list.append((df_deg_level.iloc[:, j]==1).sum()/df_deg_level.shape[0])
                not_first_prob_list.append(df_deg_level[df_deg_level.iloc[:, j]==1]['not_first_prob'].mean())

            matrix_perc.append(perc_list)
            matrix_prob.append(not_first_prob_list)
            num_data_per_level.append(df_deg_level.shape[0])

        return np.array(matrix_perc),np.array(matrix_prob), degradation_intervals, num_data_per_level, df.columns[3:df_deg_level.shape[1]-1].str.replace('rank_', '', regex=False).tolist()

    def MAEB_matrix_best_criteria_by_time(self,process_ids,train_grid_n_ep,test_grid_n_ep,test_grid_freq,global_deg_metric,local_deg_metric):
        df=self.MAEB_generate_criteria_rank_by_time_data(process_ids,train_grid_n_ep,test_grid_n_ep,test_grid_freq,global_deg_metric,local_deg_metric)
        df=df[df['process_id'].isin(process_ids)]

        deg_intervals=list(np.arange(0,1+1/3,1/3))[::-1]
        matrix_deg_intervals=[] # Aqui se almacenaran "las distribuciones" de los niveles de degradacion por intervalo de tamaños de secuencia
        
        seq_size_intervals=list(np.arange(0,1.25,0.25))
        max_size=self.iter_max-self.start_iter
        matrix_perc=[] # Aqui se almacenaran los datos para las barras apiladas


        split_seq_size=int(max_size*seq_size_intervals[1])
        for i in range(1,len(seq_size_intervals)):
            max_seq_size=split_seq_size*i
            min_seq_size=1+split_seq_size*(i-1)
            df_seq_size=df[(df['seq_size']>=min_seq_size) & (df['seq_size']<max_seq_size) ]
            perc_list=[]
            for j in range(3,df_seq_size.shape[1]-2):
                perc_list.append((df_seq_size.iloc[:, j]==1).sum()/df_seq_size.shape[0])

            provisional=list(df_seq_size['degradation_level'])            
            matrix_deg_intervals.append([ sum((deg_intervals[i] <= deg <= deg_intervals[i-1] if i == 1 else deg_intervals[i] <= deg < deg_intervals[i-1]) for deg in df_seq_size['degradation_level'] )  for i in range(1,len(deg_intervals))])
            matrix_perc.append(perc_list)


        return np.array(matrix_perc),np.array(matrix_deg_intervals),seq_size_intervals, ['['+str(round(deg_intervals[i],1))+','+str(round(deg_intervals[i-1],1))+')' for i in range(1,len(deg_intervals))], df

    def MAEB2_matrix_best_criteria_by_time(self,process_ids,train_grid_n_ep,test_grid_n_ep,test_grid_freq,global_deg_metric,local_deg_metric):
        df=self.MAEB2_generate_criteria_rank_by_time_data(process_ids,train_grid_n_ep,test_grid_n_ep,test_grid_freq,global_deg_metric,local_deg_metric)
        df=df[df['process_id'].isin(process_ids)]

        seq_size_intervals=list(np.arange(0,1.25,0.25))
        max_size=self.iter_max-self.start_iter

        # Aqui se almacenaran los datos para las barras apiladas
        matrix_perc_last_train,matrix_diff_last_train,matrix_eff_last_train=[],[[],[]],[[],[]]
        matrix_perc_last_test,matrix_diff_last_test,matrix_eff_last_test=[],[[],[]],[[],[]]
        matrix_perc_train_test,matrix_diff_train_test,matrix_eff_train_test=[],[[],[]],[[],[]]

        deg_intervals=list(np.arange(0,1+1/3,1/3))[::-1]
        matrix_deg_intervals=[] # Aqui se almacenaran "las distribuciones" de los niveles de degradacion por intervalo de tamaños de secuencia

        split_seq_size=int(max_size*seq_size_intervals[1])

        column_name=df.columns.tolist()
        for i in range(1,len(seq_size_intervals)):
            max_seq_size=split_seq_size*i
            min_seq_size=1+split_seq_size*(i-1)
            df_seq_size=df[(df['seq_size']>=min_seq_size) & (df['seq_size']<max_seq_size) ]

            # Last vs train
            last_perc=(df_seq_size.iloc[:, 3]<df_seq_size.iloc[:, 4]).sum()/df_seq_size.shape[0]
            matrix_perc_last_train.append([last_perc,1-last_perc])

            idx_best_last=df_seq_size.iloc[:, 3]<df_seq_size.iloc[:, 4]
            matrix_diff_last_train[0]+=Converter.from_lists_best_worst_to_relative_diff(df_seq_size.loc[idx_best_last,column_name[3]],df_seq_size.loc[idx_best_last,column_name[4]])
            matrix_eff_last_train[0]+=list(df_seq_size.loc[idx_best_last,column_name[6]])
            idx_best_train=df_seq_size.iloc[:, 3]>df_seq_size.iloc[:, 4]
            matrix_diff_last_train[1]+=Converter.from_lists_best_worst_to_relative_diff(df_seq_size.loc[idx_best_train,column_name[4]],df_seq_size.loc[idx_best_train,column_name[3]])
            matrix_eff_last_train[1]+=list(df_seq_size.loc[idx_best_train,column_name[7]])


            # Last vs test
            last_perc=(df_seq_size.iloc[:, 3]<df_seq_size.iloc[:, 5]).sum()/df_seq_size.shape[0]
            matrix_perc_last_test.append([last_perc,1-last_perc])

            idx_best_last=df_seq_size.iloc[:, 3]<df_seq_size.iloc[:, 5]
            matrix_diff_last_test[0]+=Converter.from_lists_best_worst_to_relative_diff(df_seq_size.loc[idx_best_last,column_name[3]],df_seq_size.loc[idx_best_last,column_name[5]])
            matrix_eff_last_test[0]+=list(df_seq_size.loc[idx_best_last,column_name[6]])
            idx_best_test=df_seq_size.iloc[:, 3]>df_seq_size.iloc[:, 5]
            matrix_diff_last_test[1]+=Converter.from_lists_best_worst_to_relative_diff(df_seq_size.loc[idx_best_test,column_name[5]],df_seq_size.loc[idx_best_test,column_name[3]])
            matrix_eff_last_test[1]+=list(df_seq_size.loc[idx_best_test,column_name[8]])

            # train vs test
            train_perc=(df_seq_size.iloc[:, 4]<df_seq_size.iloc[:, 5]).sum()/df_seq_size.shape[0]
            matrix_perc_train_test.append([train_perc,1-train_perc])

            idx_best_train=df_seq_size.iloc[:, 4]<df_seq_size.iloc[:, 5]
            matrix_diff_train_test[0]+=Converter.from_lists_best_worst_to_relative_diff(df_seq_size.loc[idx_best_train,column_name[4]],df_seq_size.loc[idx_best_train,column_name[5]])
            matrix_eff_train_test[0]+=list(df_seq_size.loc[idx_best_train,column_name[7]])
            idx_best_test=df_seq_size.iloc[:, 4]>df_seq_size.iloc[:, 5]
            matrix_diff_train_test[1]+=Converter.from_lists_best_worst_to_relative_diff(df_seq_size.loc[idx_best_test,column_name[5]],df_seq_size.loc[idx_best_test,column_name[4]])
            matrix_eff_train_test[1]+=list(df_seq_size.loc[idx_best_test,column_name[8]])

            matrix_deg_intervals.append([ sum((deg_intervals[i] <= deg <= deg_intervals[i-1] if i == 1 else deg_intervals[i] <= deg < deg_intervals[i-1]) for deg in df_seq_size['degradation_level'] )  for i in range(1,len(deg_intervals))])
    
        return np.array(matrix_perc_last_train),np.array(matrix_perc_last_test),np.array(matrix_perc_train_test),matrix_diff_last_train,matrix_diff_last_test,matrix_diff_train_test,matrix_eff_last_train,matrix_eff_last_test,matrix_eff_train_test,np.array(matrix_deg_intervals), ['['+str(round(deg_intervals[i],1))+','+str(round(deg_intervals[i-1],1))+')' for i in range(1,len(deg_intervals))], df
 
    
    def graph_best_criteria_by_time(self,process_ids,train_grid_n_ep,test_grid_n_ep,test_grid_freq,global_deg_metric='best_last_deg',local_deg_metric='paired_diff_median'):

        # Generar matriz numerica para la grafica
        matrix, matrix_prob,matrix_deg_first,matrix_deg,matrix_mag,seq_size_intervals,degradation_intervals,deg_labels,criteria_labels=self.matrix_best_criteria_by_time(process_ids,train_grid_n_ep,test_grid_n_ep,test_grid_freq,global_deg_metric,local_deg_metric)
        fig, axes = plt.subplots(2,1,figsize=(10,7),gridspec_kw={'height_ratios': [1, 3]})
        plt.subplots_adjust(left=0.08,bottom=0.33,right=0.97,top=0.92,wspace=0.39,hspace=0.07)

        # Dibujar grafica que representa distribucion de nivel de degradacion en cada intervalo de tiempo
        ax=list(axes)[0]
        colors=[Converter.generate_colormap(value,'Greys_r',0,1) for value in list(np.arange(0,1,0.2))]

        bottoms = np.zeros(len(matrix_deg))  # Inicializar las posiciones base para apilar las barras
        for i in range(len(deg_labels)):
            ax.bar(np.arange(len(matrix_deg)), matrix_deg[:, i], bottom=bottoms, label=deg_labels[i], width=1,color=colors[i])
            bottoms += matrix_deg[:, i] # Actualizar las bases para apilar la siguiente barra
        ax.set_xlim([-0.5,matrix_deg.shape[0]-.5])
        ax.set_xticks([])
        ax.set_ylabel('Number of\nsequences')
        ax.legend(title='Degradation level',loc='upper center', bbox_to_anchor=(0.1, -3.7))

        # Dibujar barras apiladas que representan mejor criterio por tiempo/tamaño de secuencia
        ax=list(axes)[1]
        colors=list(mcolors.TABLEAU_COLORS.keys())+['#fbf812','#20fa03','#0d099b','#9b094c']

        bottoms = np.zeros(len(matrix))  # Inicializar las posiciones base para apilar las barras
        for i in range(len(criteria_labels)):
            ax.bar(np.arange(len(matrix)), matrix[:, i], bottom=bottoms, label=criteria_labels[i], width=1,color=colors[i])

            # Escribir los valores dentro de las barras
            for j in range(len(matrix)):
                if not math.isnan(matrix_prob[j][i]):
                    ax.text(j, bottoms[j] + matrix[:, i][j] / 2, str((round(matrix_prob[j][i],1),round(matrix_mag[j][i],1),round(matrix_deg_first[j][i],1))),ha='center', va='center',fontsize=6, color='black',rotation=0)

            # Actualizar las bases para apilar la siguiente barra
            bottoms += matrix[:, i] 

        
        ax.set_xticks(np.arange(len(matrix)+1)-.5)  # Poner los ticks en el inicio de cada barra
        ax.set_xlim([-.5,max(np.arange(len(matrix)+1)-.5)])
        ax.set_xticklabels([str(int(i*100)) for i in seq_size_intervals], rotation=0) 
        ax.set_xlabel('Percentage of total elapsed learning iterations')
        ax.set_ylabel('Cumulative percentage of\ntimes criterion is the best')
        ax.legend(title='Criterion (when train n_ep; when test n_ep_freq)\nTuple (n1,n2,n3) inside:\n    n1: mean magnitude similarity (in [0,1]) with respect to lower ranks\n    n2: mean magnitude (in [0,1])\n    n3: mean degradation level (in [0,1])',loc='upper center', bbox_to_anchor=(0.6, -0.175), ncol=6)

        plt.savefig('experiments_intuition/results/CriteriaComparison/figures/best_criteria_by_time/'+global_deg_metric+'_'+local_deg_metric+'.pdf')
        plt.show()

    def graph_best_criteria_by_degradation(self,process_ids,title,global_deg_metric='mean_update_deg',local_deg_metric='greater_prob'):

        # Generar matriz numerica para la grafica
        matrix, matrix_prob,degradation_intervals,num_data_per_level=self.matrix_best_criteria_by_degradation(process_ids,global_deg_metric,local_deg_metric)
        fig, axes = plt.subplots(2,1,figsize=(10,5),gridspec_kw={'height_ratios': [1, 3]})
        plt.subplots_adjust(left=0.08,bottom=0.27,right=0.97,top=0.92,wspace=0.39,hspace=0.07)

        # Dibujar las barras que indican el numero de datos usados para calcular los datos de las barras apiladas de abajo
        ax=list(axes)[0]
        ax.bar(np.arange(len(matrix)), num_data_per_level, color='grey', width=1)
        ax.set_ylabel("Number of data\nper interval")
        ax.set_yscale('log')
        ax.set_xlim([-.5,len(degradation_intervals)-1.5])
        ax.set_xticks([])
        ax.set_title(title)


        # Dibujar las barras apiladas
        ax=list(axes)[1]
        colors=list(mcolors.TABLEAU_COLORS.keys())[2:]
        labels=['Last','Best train', 'Best test']

        bottoms = np.zeros(len(matrix))  # Inicializar las posiciones base para apilar las barras
        for i in range(3):
            ax.bar(np.arange(len(matrix)), matrix[:, i], bottom=bottoms, color=colors[i], label=labels[i], width=1)

            # Escribir los valores dentro de las barras
            for j in range(len(matrix)):
                if not math.isnan(matrix_prob[j][i]):
                    ax.text(j, bottoms[j] + matrix[:, i][j] / 2, str(round(matrix_prob[j][i],1)),ha='center', va='center',fontsize=6, color='black',rotation=0)

            # Actualizar las bases para apilar la siguiente barra
            bottoms += matrix[:, i] 

        
        ax.set_xticks(np.arange(len(matrix)+1)-.5)  # Poner los ticks en el inicio de cada barra
        ax.set_xlim([-.5,max(np.arange(len(matrix)+1)-.5)])
        ax.set_xticklabels([str(round(i,2)) for i in degradation_intervals], rotation=90) 
        ax.set_xlabel('Degradation level')
        ax.set_ylabel('Cumulative percentage of\ntimes criterion is the best')
        ax.legend(title='Criterion\nNumber inside: magnitude similarity (in [0,1]) with respect to lower ranks',loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3)

        plt.savefig('experiments_intuition/results/CriteriaComparison/figures/best_criteria_by_deg/'+global_deg_metric+'_'+local_deg_metric+'/advanced_'+title+'.pdf')
        #plt.show()

    def graph_best_train_test_criteria_by_degradation(self,process_ids,title,train_or_test,global_deg_metric='mean_update_deg',local_deg_metric='greater_prob'):
        # Generar matriz numerica para la grafica
        matrix, matrix_prob,degradation_intervals,num_data_per_level,labels=self.matrix_train_test_criteria_by_degradation(process_ids,global_deg_metric,local_deg_metric,train_or_test)
        fig, axes = plt.subplots(2,1,figsize=(10,5),gridspec_kw={'height_ratios': [1, 3]})
        plt.subplots_adjust(left=0.08,bottom=0.27,right=0.97,top=0.92,wspace=0.39,hspace=0.07)

        # Dibujar las barras que indican el numero de datos usados para calcular los datos de las barras apiladas de abajo
        ax=list(axes)[0]
        ax.bar(np.arange(len(matrix)), num_data_per_level, color='grey', width=1)
        ax.set_ylabel("Number of data\nper interval")
        ax.set_yscale('log')
        ax.set_xlim([-.5,len(degradation_intervals)-1.5])
        ax.set_xticks([])
        ax.set_title(title)

        # Dibujar las barras apiladas
        ax=list(axes)[1]

        bottoms = np.zeros(len(matrix))  # Inicializar las posiciones base para apilar las barras
        for i in range(len(labels)):
            ax.bar(np.arange(len(matrix)), matrix[:, i], bottom=bottoms, label=labels[i], width=1)

            # Escribir los valores dentro de las barras
            for j in range(len(matrix)):
                if not math.isnan(matrix_prob[j][i]):
                    ax.text(j, bottoms[j] + matrix[:, i][j] / 2, str(round(matrix_prob[j][i],1)),ha='center', va='center',fontsize=6, color='black',rotation=0)

            # Actualizar las bases para apilar la siguiente barra
            bottoms += matrix[:, i] 

        
        ax.set_xticks(np.arange(len(matrix)+1)-.5)  # Poner los ticks en el inicio de cada barra
        ax.set_xlim([-.5,max(np.arange(len(matrix)+1)-.5)])
        ax.set_xticklabels([str(round(i,2)) for i in degradation_intervals], rotation=90) 
        ax.set_xlabel('Degradation level')
        ax.set_ylabel('Cumulative percentage of\ntimes criterion is the best')
        ax.legend(title=train_or_test+' configuration (n_ep_freq)\nNumber inside: magnitude similarity (in [0,1]) with respect to lower ranks',loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=len(labels))

        plt.savefig('experiments_intuition/results/CriteriaComparison/figures/best_criteria_by_deg/'+global_deg_metric+'_'+local_deg_metric+'/sensitivity_'+train_or_test+'_'+title+'.pdf')
        #plt.show()

    def graph_gain_intuition_best_criteria_by_degradation(self,process_ids,grid_n_ep,grid_freq,grid_time_perc,global_deg_metric='mean_update_deg',local_deg_metric='greater_prob'):

        # Crear la figura principal y definir la cuadricula principal (sin height_ratios, solo posiciones generales)
        fig = plt.figure(figsize=(24, 30))
        outer_grid = gridspec.GridSpec(2+len(grid_freq), len(grid_time_perc), figure=fig, hspace=0.3)  

        def single_graph(outer_grid_ax,num_data_per_level,matrix,degradation_intervals,matrix_prob,labels,x_title,y_title,legend=False):
            list_outer_grid_ax = gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios=[1, 3], subplot_spec=outer_grid_ax)
            
            # Dibujar las barras que indican el numero de datos usados para calcular los datos de las barras apiladas de abajo
            ax= fig.add_subplot(list_outer_grid_ax[0]) 

            ax.bar(np.arange(len(matrix)), num_data_per_level, color='grey', width=1)
            #ax.set_ylabel("Number of data\nper interval")
            #ax.set_yscale('log')
            ax.set_xlim([-.5,len(degradation_intervals)-1.5])
            ax.set_xticks([])
            #ax.set_title(title)

            ax.set_title(x_title)

            # Dibujar las barras apiladas
            ax= fig.add_subplot(list_outer_grid_ax[1]) 

            bottoms = np.zeros(len(matrix))  # Inicializar las posiciones base para apilar las barras
            for i in range(len(labels)):
                ax.bar(np.arange(len(matrix)), matrix[:, i], bottom=bottoms, label=labels[i], width=1)

                # Escribir los valores dentro de las barras
                for j in range(len(matrix)):
                    if not math.isnan(matrix_prob[j][i]):
                        ax.text(j, bottoms[j] + matrix[:, i][j] / 2, str(round(matrix_prob[j][i],1)),ha='center', va='center',fontsize=6, color='black',rotation=0)

                # Actualizar las bases para apilar la siguiente barra
                bottoms += matrix[:, i] 

            
            ax.set_xticks(np.arange(len(matrix)+1)-.5)  # Poner los ticks en el inicio de cada barra
            ax.set_xlim([-.5,max(np.arange(len(matrix)+1)-.5)])
            ax.set_xticklabels([str(round(i,2)) for i in degradation_intervals], rotation=90) 
            #ax.set_xlabel('Degradation level')
            #ax.set_ylabel('Cumulative percentage of\ntimes criterion is the best')
            if legend:
                ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

            ax.set_ylabel(y_title)

        seq_sizes=[int((self.iter_max-self.start_iter)*perc) for perc in grid_time_perc] # Esto diferenciara las columnas
        for i in range(len(seq_sizes)):
            if i==0:
                y_titles=['Train', 'Test freq=1\n(without penalty)']+['Test freq='+str(freq) for freq in grid_freq]
            else:
                y_titles=['']*(2+len(grid_freq))

            if i==len(seq_sizes)-1:
                legend=True
            else:
                legend=False


            # Primera fila: graficas de train
            matrix, matrix_prob,degradation_intervals,num_data_per_level,labels=self.matrix_train_test_grid_by_degradation(process_ids,global_deg_metric,local_deg_metric,seq_sizes[i],'train',grid_n_ep)
            single_graph(outer_grid[i],num_data_per_level,matrix,degradation_intervals,matrix_prob,labels,'Sequences of size: <='+str(int(grid_time_perc[i]*100))+'%',y_titles[0],legend)

            # Segunda fila: graficas de test sin contar el extra de tiempo
            matrix, matrix_prob,degradation_intervals,num_data_per_level,labels=self.matrix_train_test_grid_by_degradation(process_ids,global_deg_metric,local_deg_metric,seq_sizes[i],'test',grid_n_ep)
            single_graph(outer_grid[i+len(seq_sizes)],num_data_per_level,matrix,degradation_intervals,matrix_prob,labels,'',y_titles[1],legend)

            # Resto de filas: graficas de test con frecuencia concreta por fila y contando el extra de tiempo
            for j in range(len(grid_freq)):
                matrix, matrix_prob,degradation_intervals,num_data_per_level,labels=self.matrix_train_test_grid_by_degradation(process_ids,global_deg_metric,local_deg_metric,seq_sizes[i],'test',grid_n_ep,grid_freq[j])
                single_graph(outer_grid[i+(j+2)*len(seq_sizes)],num_data_per_level,matrix,degradation_intervals,matrix_prob,labels,'',y_titles[2+j],legend)

        plt.savefig('experiments_intuition/results/CriteriaComparison/figures/best_criteria_by_deg/'+global_deg_metric+'_'+local_deg_metric+'/gaining_intuition.pdf')
        #plt.show()

    def MAEB_graph_best_criteria_by_time(self,title,process_ids,train_grid_n_ep,test_grid_n_ep,test_grid_freq,global_deg_metric='best_last_deg',local_deg_metric='paired_diff_probpos',MAEB=False):

        plt.rc('font', family='serif',size=15)
        plt.rc('text', usetex=True)

        # Generar matriz numerica para la grafica
        matrix,matrix_deg,seq_size_intervals,deg_labels,df=self.MAEB_matrix_best_criteria_by_time(process_ids,train_grid_n_ep,test_grid_n_ep,test_grid_freq,global_deg_metric,local_deg_metric)
        fig, axes = plt.subplots(2,4,figsize=(14,5),gridspec_kw={'height_ratios': [1, 3]})
        plt.subplots_adjust(left=0.07,bottom=0.15,right=0.99,top=0.85,wspace=0.33,hspace=0.13)

        # 1) Distribucion de nivel de degradacion en cada intervalo de tiempo
        ax=axes[0,0]
        colors=[Converter.generate_colormap(round(value,2),'Greys_r',0,1) for value in [0.3,0.55,0.8]]
        deg_types=['Catastrófico','Crítico','Moderado']
        criteria_labels=['Last','Train','Test']

        bottoms = np.zeros(len(matrix_deg))  # Inicializar las posiciones base para apilar las barras
        for i in range(len(deg_labels)):
            ax.bar(np.arange(len(matrix_deg)), matrix_deg[:, i], bottom=bottoms, label=deg_types[i], width=1,color=colors[i])
            bottoms += matrix_deg[:, i] # Actualizar las bases para apilar la siguiente barra
        ax.set_xlim([-0.5,matrix_deg.shape[0]-.5])
        ax.set_xticks([])
        ax.set_ylabel("Número de\nsecuencias")#"Number of\nsequences")
        ax.legend(title='Nivel de degradación',loc='upper center', bbox_to_anchor=(1.5, 1.9))
        #ax.set_ylim([0,2000])

        # 2) Mejor criterio por tiempo/tamaño de secuencia
        ax=axes[1,0]
        colors=list(mcolors.TABLEAU_COLORS.keys())+['#fbf812','#20fa03','#0d099b','#9b094c']
        
        bottoms = np.zeros(len(matrix))  # Inicializar las posiciones base para apilar las barras
        for i in range(len(criteria_labels)):
            ax.bar(np.arange(len(matrix)), matrix[:, i], bottom=bottoms, label=criteria_labels[i], width=1,color=colors[i])
            bottoms += matrix[:, i] 

        ax.set_xticks(np.arange(len(matrix)+1)-.5)  # Poner los ticks en el inicio de cada barra
        ax.set_xlim([-.5,max(np.arange(len(matrix)+1)-.5)])
        ax.set_xticklabels([str(int(i*100)) for i in seq_size_intervals], rotation=0) 
        ax.set_xlabel('Tiempo de aprendizaje ($t$)')
        ax.set_ylabel('Proporción de mejor criterio')
        ax.legend(title='Criterion',loc='upper center', bbox_to_anchor=(2.3, 1.7), ncol=1)

        # 3) Distribuciones de degradacion para cada criterio (cuando es cada criterio el mejor)
        axes[0,1].set_visible(False)
        ax = axes[1,1]

        deg_by_criteria_when_best=[]
        colum_names=df.columns.tolist()
        for i in [3,4,5]:
            deg_by_criteria_when_best.append(list(df[df[colum_names[i]]==1]['degradation_level']))

        colors_grey=[Converter.generate_colormap(round(value,2),'Greys_r',0,1) for value in [0.3,0.55,0.8]]
        ax.axhspan(0, 0.3333, facecolor=colors_grey[2])
        ax.axhspan(0.3333, 0.6666, facecolor=colors_grey[1])
        ax.axhspan(0.6666, 1, facecolor=colors_grey[0])
        vp = ax.violinplot(deg_by_criteria_when_best, showmeans=False, showmedians=False, showextrema=False)
        for i, body in enumerate(vp['bodies']): # Asignar colores a cada violín
            body.set_facecolor(colors[i])
            body.set_alpha(1)

        ax.set_xticks(range(1, len(criteria_labels) + 1))
        ax.set_xticklabels(criteria_labels)
        ax.set_ylim(0, 1)
        ax.set_title('When is it the best?')
        ax.set_ylabel('Nivel de degradación')

        # 4) Por cuanta diferencia es el mejor
        axes[0,2].set_visible(False)
        ax = axes[1,2]

        diff_by_criteria_when_best=[]
        for i in [3,4,5]:
            diff_by_criteria_when_best.append(list(df[df[colum_names[i]]==1]['not_first_prob']))

        vp = ax.violinplot(diff_by_criteria_when_best, showmeans=False, showmedians=False, showextrema=False)
        for i, body in enumerate(vp['bodies']): # Asignar colores a cada violín
            body.set_facecolor(colors[i])
            body.set_alpha(1)

        ax.set_xticks(range(1, len(criteria_labels) + 1))
        ax.set_xticklabels(criteria_labels)
        ax.set_ylim(0, 1)
        ax.set_title('How much better is it\nthan the rest?')
        ax.set_ylabel("Percentage of magnitude\ncompared to next best")

        # 5) Es mejor, pero es bueno
        axes[0,3].set_visible(False)
        ax = axes[1,3]

        eff_by_criteria_when_best=[]
        for i in [3,4,5]:
            eff_by_criteria_when_best.append(list(df[df[colum_names[i]]==1]['first_eff']))

        vp = ax.violinplot(eff_by_criteria_when_best, showmeans=False, showmedians=False, showextrema=False)
        for i, body in enumerate(vp['bodies']): # Asignar colores a cada violín
            body.set_facecolor(colors[i])
            body.set_alpha(1)

        ax.set_xticks(range(1, len(criteria_labels) + 1))
        ax.set_xticklabels(criteria_labels)
        ax.set_ylim(0, 1)
        ax.set_title('How good is it selecting?')
        ax.set_ylabel("Percentage of objective value\ncompared to the real best")

        if MAEB:
            plt.savefig('experiments_intuition/results/MAEB/'+global_deg_metric+'_'+local_deg_metric+'_'+title+'.pdf')
        
            # OTRO: distribucion de la degradacion con la metrica propuesta
            plt.figure(figsize=[5,2.5])
            plt.subplots_adjust(left=0.17,bottom=0.21,right=0.94,top=0.82,wspace=0.21,hspace=0.2)
            sns.kdeplot(df['degradation_level'], bw_adjust=0.5, clip=(0, 1), color='black', fill=True, alpha=0.4)
            plt.xlabel('Nivel de degradación')
            plt.ylabel('Densidad')
            plt.savefig('experiments_intuition/results/MAEB/deg_distribution.pdf')

            # OTRO: una learning-curve por cada caso de degradacion para ilustrar la discretizacion
            def deg_curve(df_dreg,title):
                row_max= df.loc[df_dreg['degradation_level'].idxmax()]
                current_path=parent_dir+'/_bender/project_SB3/data/'+row_max['process_id']+'_16cpu1gpu_mejorado'
                df_test_estimates=pd.read_csv(current_path+'/df_val_estimates.csv')
                valores=list(df_test_estimates['truth_norm'])[self.start_iter:self.start_iter+row_max['seq_size']]
                print('deg level:', row_max['degradation_level'])
                print('process_id:',row_max['process_id'])

                fig=plt.figure(figsize=[5,2.5])
                plt.subplots_adjust(left=0.17,bottom=0.21,right=0.94,top=0.82,wspace=0.21,hspace=0.2)
                ax=plt.subplot(111)
                plt.plot(list(range(len(valores))), valores, linewidth=1,color='black')
                plt.title(title)
                ax.set_xlabel("Iteraciones de aprendizaje")
                ax.set_ylabel("$\widetilde{f}(\pi_t)$")
                plt.savefig('experiments_intuition/results/MAEB/deg_'+title+'.pdf')
                plt.show()

            df_catastrophic=df[(df['degradation_level']>0.9) & (df['degradation_level']<0.95)]# 47 (catastrofico)
            deg_curve(df_catastrophic,'Catastrófico')
            df_critic=df[df['degradation_level']==0.658]#220 (critico)
            deg_curve(df_critic,'Crítico')
            df_moderate=df[df['degradation_level']==0.206]#276 (moderado)
            deg_curve(df_moderate,'Moderado')

        else:
            plt.savefig('experiments_intuition/results/CriteriaComparison/figures/best_criteria_by_time_related_deg/'+global_deg_metric+'_'+local_deg_metric+'_'+title+'.pdf')
        
        

    def MAEB2_graph_best_criteria_by_time(self,title,process_ids,train_grid_n_ep,test_grid_n_ep,test_grid_freq,global_deg_metric='best_last_deg',local_deg_metric='paired_diff_probpos',MAEB=False):
        
        plt.rc('font', family='serif')
        plt.rc('text', usetex=True)


        # Generar matriz numerica para la grafica
        matrix_last_train,matrix_last_test,matrix_train_test,matrix_diff_last_train,matrix_diff_last_test,matrix_diff_train_test,matrix_eff_last_train,matrix_eff_last_test,matrix_eff_train_test,matrix_deg,deg_labels,df=self.MAEB2_matrix_best_criteria_by_time(process_ids,train_grid_n_ep,test_grid_n_ep,test_grid_freq,global_deg_metric,local_deg_metric)

        fig, axes = plt.subplots(4,4,figsize=(9,7),gridspec_kw={'height_ratios': [1,3,3,3]})
        plt.subplots_adjust(left=0.1,bottom=0.08,right=0.97,top=0.95,wspace=0.39,hspace=0.42)
        
        # Dibujar grafica que representa distribucion de nivel de degradacion en cada intervalo de tiempo
        ax=axes[0,0]
        colors=[Converter.generate_colormap(round(value,2),'Greys_r',0,1) for value in [0.3,0.55,0.8]]
        deg_types=['Catastrophic','Critic','Moderate']
        criteria_labels=['Last','Train','Test']

        bottoms = np.zeros(len(matrix_deg))  # Inicializar las posiciones base para apilar las barras
        for i in range(len(deg_labels)):
            ax.bar(np.arange(len(matrix_deg)), matrix_deg[:, i], bottom=bottoms, label=deg_labels[i]+' '+deg_types[i], width=1,color=colors[i])
            bottoms += matrix_deg[:, i] # Actualizar las bases para apilar la siguiente barra
        ax.set_xlim([-0.5,matrix_deg.shape[0]-.5])
        ax.set_xticks([])

        axes[0,1].set_visible(False)
        axes[0,2].set_visible(False)
        axes[0,3].set_visible(False)

        # Grafica de barras
        def barplots_2criteria(ax,matrix,colors,labels):
            bottoms = np.zeros(len(matrix))  # Inicializar las posiciones base para apilar las barras
            for i in range(2):
                ax.bar(np.arange(len(matrix)), matrix[:, i], bottom=bottoms, label=labels[i], width=1,color=colors[i])
                bottoms += matrix[:, i] 
            ax.set_xticks(np.arange(len(matrix)+1)-.5)  # Poner los ticks en el inicio de cada barra
            ax.set_xlim([-.5,max(np.arange(len(matrix)+1)-.5)])
            ax.set_xticklabels([str(int(i*100)) for i in list(np.arange(0,1.25,0.25))], rotation=0) 

        colors=list(mcolors.TABLEAU_COLORS.keys())+['#fbf812','#20fa03','#0d099b','#9b094c']
        barplots_2criteria(axes[1,0],matrix_last_train,[colors[0],colors[1]],[criteria_labels[0],criteria_labels[1]])
        barplots_2criteria(axes[2,0],matrix_last_test,[colors[0],colors[2]],[criteria_labels[0],criteria_labels[2]])
        barplots_2criteria(axes[3,0],matrix_train_test,[colors[1],colors[2]],[criteria_labels[1],criteria_labels[2]])

        axes[1,0].set_title('How many times is each\ncriterion the best')
        axes[3,0].set_xlabel("Percentage of total elapsed\nlearning iterations")

        # Gafica de degradaciones
        def violinplot_deg_2criteria(ax,criteria_columns,colors,labels):
            
            deg_by_criteria_when_best=[]
            colum_names=df.columns.tolist()
            deg_by_criteria_when_best.append(list(df[df[colum_names[criteria_columns[0]]]<df[colum_names[criteria_columns[1]]]]['degradation_level']))
            deg_by_criteria_when_best.append(list(df[df[colum_names[criteria_columns[1]]]<df[colum_names[criteria_columns[0]]]]['degradation_level']))

            # Añadir franjas horizontales (rangos de degradacion)
            colors_grey=[Converter.generate_colormap(round(value,2),'Greys_r',0,1) for value in [0.3,0.55,0.8]]
            ax.axhspan(0, 0.3333, facecolor=colors_grey[2])
            ax.axhspan(0.3333, 0.6666, facecolor=colors_grey[1])
            ax.axhspan(0.6666, 1, facecolor=colors_grey[0])
        

            # Crear violin plots
            vp = ax.violinplot(deg_by_criteria_when_best, showmeans=False, showmedians=False, showextrema=False)
            for i, body in enumerate(vp['bodies']): # Asignar colores a cada violín
                body.set_facecolor(colors[i])
                body.set_edgecolor(colors[i])
                body.set_alpha(1)
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels)


        colors=list(mcolors.TABLEAU_COLORS.keys())+['#fbf812','#20fa03','#0d099b','#9b094c']
        violinplot_deg_2criteria(axes[1,1],[3,4],[colors[0],colors[1]],[criteria_labels[0],criteria_labels[1]])
        violinplot_deg_2criteria(axes[2,1],[3,5],[colors[0],colors[2]],[criteria_labels[0],criteria_labels[2]])
        violinplot_deg_2criteria(axes[3,1],[4,5],[colors[1],colors[2]],[criteria_labels[1],criteria_labels[2]])
        
        axes[1,1].set_title('When is it the best?\n')

        # Grafica de diferencias
        def violinplot_2criteria(ax,data,colors,labels):

            # Crear violin plots
            vp = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
            for i, body in enumerate(vp['bodies']): # Asignar colores a cada violín
                body.set_facecolor(colors[i])
                body.set_edgecolor(colors[i])
                body.set_alpha(1)
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels)


        colors=list(mcolors.TABLEAU_COLORS.keys())+['#fbf812','#20fa03','#0d099b','#9b094c']
        violinplot_2criteria(axes[1,2],matrix_diff_last_train,[colors[0],colors[1]],[criteria_labels[0],criteria_labels[1]])
        violinplot_2criteria(axes[2,2],matrix_diff_last_test,[colors[0],colors[2]],[criteria_labels[0],criteria_labels[2]])
        violinplot_2criteria(axes[3,2],matrix_diff_train_test,[colors[1],colors[2]],[criteria_labels[1],criteria_labels[2]])
        violinplot_2criteria(axes[1,3],matrix_eff_last_train,[colors[0],colors[1]],[criteria_labels[0],criteria_labels[1]])
        violinplot_2criteria(axes[2,3],matrix_eff_last_test,[colors[0],colors[2]],[criteria_labels[0],criteria_labels[2]])
        violinplot_2criteria(axes[3,3],matrix_eff_train_test,[colors[1],colors[2]],[criteria_labels[1],criteria_labels[2]])

        axes[1,2].set_title('How match better is it\nthan the other?')
        axes[1,3].set_title('How good is it selecting?\n')

        if MAEB:
            plt.savefig('experiments_intuition/results/MAEB/vs2_'+global_deg_metric+'_'+local_deg_metric+'_'+title+'.pdf')
        else:
            plt.savefig('experiments_intuition/results/CriteriaComparison/figures/best_criteria_by_time_related_deg/vs2_'+global_deg_metric+'_'+local_deg_metric+'_'+title+'.pdf')
        plt.show()

   
    #===============================================================================================
    # NUEVO ANALISIS
    #===============================================================================================
    # Distribucion de degradacion por entorno-tiempo (creo que esta grafica no es necesaria)
    def graph_deg_distribution_per_env_and_time(self,algo,envs,global_deg_metric,local_deg_metric):
        '''
        Distribucion de degradacion por entorno e intervalo de tiempo. Esta grafica permite comparar las degradaciones por 
        (complejidad) entorno y por tiempo invertido en aprender.
        '''
        # Reducir base de datos a la degradacion que nos interesa
        df=self.df_degradation.filter(like=global_deg_metric+'_'+local_deg_metric)

        fig = plt.figure(figsize=(8,5))
        plt.subplots_adjust(left=0.08,bottom=0.09,right=0.97,top=0.88,wspace=0.29,hspace=0.07)
        
        gs = fig.add_gridspec(4, 2, width_ratios=[2, 1])  # primera col mas ancha
        colors=list(mcolors.TABLEAU_COLORS.keys())+['#fbf812','#20fa03','#0d099b','#9b094c']
        x = np.linspace(0, 1, 200)

        # Todos los tiempos juntos
        ax = fig.add_subplot(gs[:, 0]) 
        ax.grid(True, alpha=0.3)
        plt.title('Degradation distribution by environment\nat different percentages of learning time for convergence')

        for i in range(len(envs)):
            # Generar una lista con las degradaciones acumuladas para ese algo_env
            process_id_init=algo+'_'+envs[i]            
            sample=df.filter(like=process_id_init).values.flatten()

            # KDE para esos datos
            kde= gaussian_kde(sample)
            ax.plot(x, kde(x), label=envs[i],alpha=0.6,color=colors[i],linewidth=1)
            ax.fill_between(x, kde(x), alpha=0.4,color=colors[i])
            ax.axvline(np.median(sample), linestyle="-", linewidth=1.5,color=colors[i])

        ax.set_xlabel("Degradation")
        ax.set_ylabel("KDE")
        ax.legend()

        # Dividido por intervalor de tiempo
        points=np.linspace(0, 1, 4+1)# 4 intervalos
        intervals= list(zip(points[:-1], points[1:]))

        for i,(start, end) in zip(range(4+1),intervals):
            ax = fig.add_subplot(gs[i, 1]) 

            # Subconjunto de datos
            subset = df.iloc[int(len(df) * start):int(len(df) * end)]
            
            for j in range(len(envs)):
                sample=subset.filter(like=envs[j]).values.flatten()
                kde= gaussian_kde(sample)
                ax.plot(x, kde(x), label=envs[j],alpha=0.6,color=colors[j],linewidth=1)
                ax.fill_between(x, kde(x), alpha=0.4,color=colors[j])
                ax.axvline(np.median(sample), linestyle="-", linewidth=1.5,color=colors[j])
                
            ax.set_ylabel(f"{int(start*100)}% - {int(end*100)}%")
            ax.grid(True, alpha=0.3)
            
        ax.set_xlabel("Degradation")
        plt.savefig('experiments_intuition/results/CriteriaComparison/figures/deg_distribution/'+algo+'_'+global_deg_metric+'_'+local_deg_metric+'.pdf')
        plt.show()

    # Analisis que cierra todo: ¿cuando usar que criterio?
    def generate_df_test_strat_end(self, process_id,n_ep,freq):

        algo,env,seed=Converter.process_id_splitter(process_id)
        process_id=algo+'_'+env+'_seed'+str(seed)
        test_conf=str(n_ep)+'_'+str(freq)
        generator=EvolutionGenerator(algo,env,seed,'16cpu1gpu_mejorado',perc_time_start=0.1)

        # Mirar si no se ha llamado antes a esta funcion de la misma manera
        already_registered=(self.df_test_start_end['process_id']==process_id) & (self.df_test_start_end['test_conf']==test_conf)

        df_add=[]
        all_time_seq=generator.df_train.loc[self.start_iter:self.iter_max,'time_seconds'].tolist()

        if not np.array(already_registered).any():

            # ------------Test ascendente
            starts=[0]*len(all_time_seq)
            ends=list(range(len(all_time_seq)))
            for start,end in zip(starts,ends):
                # Apilcando last
                if start==end:
                    policy_id=generator.last_policy(all_time_seq[-1])
                    best_policy_id=generator.truth_best_policy(all_time_seq[-1])
                    val_time=0

                #Aplicando test en un intervalo diferente
                else:
                    policy_id,val_time=generator.best_policy_validation(all_time_seq[end],n_ep,all_time_seq)
                    best_policy_id=generator.truth_best_policy(all_time_seq[-1])

                perc_good=generator.df_test_estimates.loc[policy_id,'truth']/generator.df_test_estimates.loc[best_policy_id,'truth']
                invest_time=val_time+all_time_seq[-1]

                #Aplicar last tambien
                policy_id=generator.last_policy(all_time_seq[end])
                best_policy_id=generator.truth_best_policy(all_time_seq[end])

                last_good=generator.df_test_estimates.loc[policy_id,'truth']/generator.df_test_estimates.loc[best_policy_id,'truth']

                df_add.append([process_id,test_conf,start,end,perc_good,invest_time,last_good,all_time_seq[end]])

            # -------------Test simetrico
            starts=list(range(int(len(all_time_seq)/2)))
            ends=[(self.iter_max-self.start_iter-1)-i for i in starts]
            for start,end in zip(starts,ends):
                #Aplicando test en un intervalo diferente
                if start!=end:
                    
                    policy_id,val_time=generator.best_policy_validation(all_time_seq[end],n_ep,all_time_seq[start:])
                    best_policy_id=generator.truth_best_policy(all_time_seq[-1])

                perc_good=generator.df_test_estimates.loc[policy_id,'truth']/generator.df_test_estimates.loc[best_policy_id,'truth']
                invest_time=val_time+all_time_seq[-1]

                df_add.append([process_id,test_conf,start,end,perc_good,invest_time,None,None])

            # Añadir nuevas filas a la base de datos existente
            df_add = pd.DataFrame(df_add, columns=self.df_test_start_end.columns)
            self.df_test_start_end = pd.concat([self.df_test_start_end, df_add],ignore_index=True)
            self.df_test_start_end.to_csv('experiments_intuition/results/CriteriaComparison/data/test_start_end_perc_time.csv', index=False)
        
    def graph_when_use_test(self,process_ids,n_ep,freq,title,global_deg_metric,local_deg_metric):

        # Generar los datos necesarios
        # for process_id in tqdm(process_ids):
        #     self.generate_df_test_strat_end(process_id,n_ep,freq)
        
        # Funciones internas de esta funcion
        def get_perc_time(process_ids,n_ep,freq,ends,test):

            #Seleccionar filas de interes
            df=self.df_test_start_end
            if test=='ascendent':
                rows=df['process_id'].isin(process_ids) & (df['test_conf']==str(n_ep)+'_'+str(freq)) & (df['start_test']==0) & (df['end_test'].isin(ends)) 
            if test=='simetric':
                rows=df['process_id'].isin(process_ids) & (df['test_conf']==str(n_ep)+'_'+str(freq)) & (df['start_test']==abs(df['end_test']-(self.iter_max-self.start_iter-1))) & (df['start_test'].isin(ends))
            
            if test=='time_conv':
                rows=df['process_id'].isin(process_ids) & (df['test_conf']==str(n_ep)+'_'+str(freq)) & (df['start_test']==0) & (df['end_test']==0) 
            if test=='only_last':
                rows=df['process_id'].isin(process_ids) & (df['test_conf']==str(n_ep)+'_'+str(freq)) & (df['start_test']==0) & (df['end_test'].isin(ends)) 
            
            if test=='only_last':
                return np.mean(df.loc[rows,'last_good']), np.mean(df.loc[rows,'last_time'])
            else:
                return np.mean(df.loc[rows,'perc_good']), np.mean(df.loc[rows,'invest_time'])
        
        def graph_ascendent_simetric(lista1,lista2,lista3,test,inner):
            axes = inner.subplots(sharey=True)

            # Grafica 1: cuando se aplica test
            axes[0].grid(True, alpha=0.3,color='black',linewidth=0.3)

            for i, (x1, x2) in enumerate(lista1):
                if test=='ascendent':
                    min_x = min(x1 for x1, _ in lista1)
                    if i==len(lista1)-1:
                        axes[0].barh(i, x2 - x1, left=x1, color="orange", alpha=0.6, height=1, linewidth=0.0 )#se aplica alguna vez test
                        axes[0].barh(i, x1 - min_x, left=min_x, color="orange", height=1)#se aplica test fijo
                    else:
                        axes[0].barh(i, x2 - x1, left=x1, color="green", alpha=0.6, height=1, linewidth=0.0 )#se aplica alguna vez test
                        axes[0].barh(i, x1 - min_x, left=min_x, color="green", height=1)#se aplica test fijo

                if test=='simetric':
                    axes[0].barh(i, x2 - x1, left=x1, color="green", alpha=0.6, height=1, linewidth=0.0 )#se aplica alguna vez test
                    axes[0].barh(i, (1-x1) - (1-x2), left=1-x2, color="green", alpha=0.6, height=1, linewidth=0.0 )#se aplica alguna vez test
                    axes[0].barh(i, (1-x2) - x2, left=x2, color="green", height=1)#se aplica test fijo

            # Grafica 2: como de buena es la politica seleccionada con respecto a la mejor
            axes[1].grid(True, alpha=0.3,color='black',linewidth=0.3)
            lista2_improv=[]
            for i, val in enumerate(lista2):
                axes[1].barh(i, -val, color="grey", height=1,alpha=0.5)
                if i<len(lista2)-1:
                    if test=='ascendent':
                        lista2_improv.append([lista2[i]-lista2[i+1] if lista2[i]-lista2[i+1]>0 else 0][0])
                    if test=='simetric':
                        lista2_improv.append([lista2[i+1]-lista2[i] if lista2[i+1]-lista2[i]>0 else 0][0])
            axes[1].xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{abs(x):.1f}"))
            axes[1].set_xlim(-1,0)
            axes[1].plot(-np.array(lista2_improv)[::-1], list(range(len(lista2)-1))[::-1], "-", color="grey")

            # Grafica 3: tiempo invertido en total
            axes[2].grid(True, alpha=0.3,color='black',linewidth=0.3)
            lista3_improv=[]
            for i, val in enumerate(lista3):
                axes[2].barh(i, val, color="red", height=1,alpha=0.5)
                if i<len(lista3)-1:
                    if test=='ascendent':
                        lista3_improv.append([lista3[i]-lista3[i+1] if lista3[i]-lista3[i+1]>0 else 0][0])
                    if test=='simetric':
                        lista3_improv.append([lista3[i+1]-lista3[i] if lista3[i+1]-lista3[i]>0 else 0][0])

            axes[2].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            axes[2].set_xlim(1,round(max(lista3),4)+(round(max(lista3),4)-1)*0.1)
               

            if test=='ascendent':
                axes[1].plot(-np.array(lista3_improv), list(range(len(lista3)-1)), "-", color="red")
                axes[0].set_title("When is test (green)\nand last (orange) applied?")
                axes[1].set_title("How good is the selected policy?")
                axes[2].set_title("How much time is required?")
                for ax in axes:
                    ax.set_xticklabels([]) 
                    ax.tick_params(axis='x')
            if test=='simetric':
                axes[1].plot(-np.array(lista3_improv), list(range(len(lista3)-1))[::-1], "-", color="red")
                axes[0].set_xlabel("Percentage of learning time\nfor convergence")
                axes[1].set_xlabel("Reward percentage of selected policy\nrelative to the best")
                axes[2].set_xlabel("Invest time percentage relative to\nthat required to converge")

            for ax in axes:
                ax.set_yticks(list(range(0,len(lista1)+2,2)))
                ax.set_yticklabels([]) 
                ax.tick_params(axis='y')
            #axes[1].xaxis.set_major_locator(MultipleLocator(0.1))
            #axes[2].xaxis.set_major_locator(MultipleLocator(0.01))

        def graph_test_application_perc_time(process_ids,test_type,outer):

            if test_type=='ascendent':
                interval=(np.linspace(0, 1, 10+1)**1).tolist() #TODO: aqui se podria definir una reparametrizacion del intervalo [0,1], para que el ritmo con que se aumenta/disminuye el start/end de test no sea lineal
            if test_type=='simetric':
                interval=(np.linspace(0, 0.5, 10+1)**1).tolist()


            _,time_conv=get_perc_time(process_ids,n_ep,freq,None,test='time_conv')
            variable_start_end=[]# Para la grafica ilustrativa de aplicacion de test
            all_perc_good,all_invest_time=[],[] # Para las graficas de perc y time

            for i in range(len(interval)-1):
                ends=range(int(interval[i]*(self.iter_max-self.start_iter)),int(interval[i+1]*(self.iter_max-self.start_iter))+1)
                perc_good,invest_time=get_perc_time(process_ids,n_ep,freq,ends,test_type)
                variable_start_end.append([ends[0]/(self.iter_max-self.start_iter),ends[-1]/(self.iter_max-self.start_iter)])
                all_perc_good.append(perc_good)
                all_invest_time.append(invest_time/time_conv)

            if test_type=='ascendent':
                variable_start_end=variable_start_end[::-1]
                all_perc_good=all_perc_good[::-1]
                all_invest_time=all_invest_time[::-1]

                ends=range(int(interval[i-1]*(self.iter_max-self.start_iter)),int(interval[i+1]*(self.iter_max-self.start_iter))+1)
                perc_good,invest_time=get_perc_time(process_ids,n_ep,freq,ends,'only_last')
                all_perc_good.append(perc_good)
                all_invest_time.append(1)
                variable_start_end.append([ends[0]/(self.iter_max-self.start_iter),ends[-1]/(self.iter_max-self.start_iter)])

                graph_ascendent_simetric(variable_start_end,all_perc_good,all_invest_time,'ascendent',
                                         outer.subgridspec(1, 3, wspace=0.1, hspace=0.3))
                
                return all_perc_good,all_invest_time, ends
            
            if test_type=='simetric':
                graph_ascendent_simetric(variable_start_end[::-1],all_perc_good[::-1],all_invest_time[::-1],'simetric',
                                         outer.subgridspec(1, 3, wspace=0.1, hspace=0.3))
                
                return all_perc_good,all_invest_time
        
        def graph_optimal_test_application(process_ids,ascen_perc_good,sim_perc_good,ascen_perc_time,sim_perc_time,
                                           ends_last,global_deg_metric,local_deg_metric,sub_title):

            # Datos de interes
            regex = "|".join(process_ids) 
            df_degradation=self.df_degradation[self.df_degradation['n_policy'].isin(ends_last)]
            df_interest_process = df_degradation.filter(regex=regex)
            df_interest_deg=df_interest_process.filter(like=global_deg_metric+'_'+local_deg_metric)
            ends_deg_distribution=df_interest_deg.values.flatten().tolist()

            fig = plt.figure(figsize=(9, 4))
            plt.subplots_adjust(left=0.07,bottom=0.15,right=0.97,top=0.88,wspace=0.2,hspace=0.19)
            gs = fig.add_gridspec(2, 2, height_ratios=[1, 3])  # fila1 ocupa 1, fila2 ocupa 3

            # Grafica 1: distribucion de degradacion en tiempos proximos a convergencia
            ax1 = fig.add_subplot(gs[0, 0])
            kde = gaussian_kde(ends_deg_distribution)
            x = np.linspace(0, 1, 200)
            ax1.fill_between(x, kde(x), color="orange") 
            ax1.set_xlabel('Degradation in times close to convergence')

            # Grafica 2: porcentage de reward de la politica seleccionada, test ascendente vs simetrico
            ax2 = fig.add_subplot(gs[1, 0])
            ax2.plot(ascen_perc_good, np.arange(len(ascen_perc_good)), color="gray", linestyle="-", label="ascendent")
            ax2.plot(sim_perc_good, np.arange(len(sim_perc_good)), color="gray", linestyle="--", label="simetric")
            ax2.plot(ascen_perc_good[-1], len(ascen_perc_good)-1, 'o', color="orange", markersize=8)

            ax2.invert_xaxis() 
            ax2.set_yticklabels([])
            ax2.grid(True, linestyle=":", alpha=0.7)
            ax2.set_title("")
            ax2.set_xlabel("Reward percentage of selected policy\nrelative to the best")
            ax2.legend(title='Test type')

            # Grafica 3: porcentage de tiempo usando test, test ascendente vs simetrico
            ax3 = fig.add_subplot(gs[1, 1])
            ax3.plot(ascen_perc_time, np.arange(len(ascen_perc_time)), color="red", linestyle="-", label="ascendent")
            ax3.plot(sim_perc_time, np.arange(len(sim_perc_time)), color="red", linestyle="--", label="simetric")

            ax3.legend(title='Test type')
            ax3.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            ax3.set_yticklabels([])
            ax3.grid(True, linestyle=":", alpha=0.7)
            ax3.set_title("")
            ax3.set_xlabel("Invest time percentage relative to\nthat required to converge")

            plt.savefig('experiments_intuition/results/CriteriaComparison/figures/when_criteria/'+title+'_'+sub_title+'_opt.pdf')
            plt.show()
        
            return df_interest_deg
            
        # Graficas
        fig = plt.figure(figsize=(13, 4))
        plt.subplots_adjust(left=0.08,bottom=0.15,right=0.97,top=0.88,wspace=0.2,hspace=0.07)

        outer = fig.add_gridspec(2,1)
        ascen_perc_good,ascen_perc_time,ends_last=graph_test_application_perc_time(process_ids,'ascendent',outer[0])
        sim_perc_good,sim_perc_time=graph_test_application_perc_time(process_ids,'simetric',outer[1])

        plt.savefig('experiments_intuition/results/CriteriaComparison/figures/conclusion_criteria_application/'+title+'.pdf')
        plt.show()

        df_interest_deg=graph_optimal_test_application(process_ids,ascen_perc_good,sim_perc_good,ascen_perc_time,sim_perc_time,
                                       ends_last,global_deg_metric,local_deg_metric,'')
        
        # Solo quedarnos con los procesos indicados que tienen la degradacion indicada en los tiempos de convergencia
        # process_ids_with_deg = df_interest_deg.columns[(df_interest_deg > 0.5).all()]
        # new_process_ids=[s for s in process_ids if any(s in t for t in process_ids_with_deg)]
        # print(new_process_ids)
        # fig = plt.figure(figsize=(13, 4))
        # plt.subplots_adjust(left=0.08,bottom=0.15,right=0.97,top=0.88,wspace=0.2,hspace=0.07)

        # outer = fig.add_gridspec(2,1)
        # ascen_perc_good,ascen_perc_time,ends_last=graph_test_application_perc_time(new_process_ids,'ascendent',outer[0])
        # sim_perc_good,sim_perc_time=graph_test_application_perc_time(new_process_ids,'simetric',outer[1])

        # plt.savefig('experiments_intuition/results/CriteriaComparison/figures/when_criteria/'+title+'_with_deg.pdf')
        # plt.show()

        # _=graph_optimal_test_application(new_process_ids,ascen_perc_good,sim_perc_good,ascen_perc_time,sim_perc_time,
        #                                ends_last,global_deg_metric,local_deg_metric,'with_deg')



class SingleProcessAnalisys():
    '''
    Esta clase ejecuta un analisis comparativo de criterios completo para un unico proceso. Para el proceso indicado
    (algortimo, environmnet, semilla, recursos), almacena en una carpeta independiente para el proceso los siguientes resultados graficos:
    1) Descripcion del proceso: learning-curve con degradation evolution
    2) Analisis de estimadores de expected epsodic reward: coste de estimacion y precision de estimadores para seleccion
    3) Ajuste de configuraciones optimas para train y test
    4) Comparacion de los tres criterios last, train y test (en sus mejores versiones/configuraciones): evoluciones de ranking y magnitudes (con comparacion de areas bajo curvas)
    '''
    def __init__(self,algo,env,seed,resources,max_iter):
            
        # Generar directorio donde se almacenaran las graficas.
        os.makedirs('experiments_intuition/results/CriteriaComparison/figures/'+algo+'_'+env+'_seed'+str(seed), exist_ok=True)
        
        # Valores de parametros a considerara (se hara el tuning entre estos)
        list_n_ep=[500,250,100,50,25,5]# Como las bases de datos "mejoradas" tienen almacenadas 1000 validaciones por politica
        list_freq=[100,50,25,10,5,1]
        
        # Representacion del proceso (considerando diferentes definiciones de metrica global y local de degradacion)
        grapher=EvolutionGrapher(algo,env,seed,resources,max_iter)
        #grapher.learning_curve()
        # grapher.graph_degradation_evolution('mean_update_deg','greater_prob')
        # grapher.graph_degradation_evolution('weighted_mean_best_later_deg','greater_prob')
        # grapher.graph_degradation_evolution('weighted_mean_best_later_deg','paired_diff_probpos_meanpos')
        # grapher.graph_degradation_evolution('best_last_deg','greater_prob')
        # grapher.graph_degradation_evolution('best_last_deg','paired_diff_median')
        # grapher.graph_degradation_evolution('best_last_deg','paired_diff_probpos')
        grapher.graph_degradation_evolution('relative_worsening_to_improvement','relative_reward_diff')
    
        # Analisis de estimaciones (coste y precision de seleccion)
        analyzer=EstimationAnalyzer(algo,env,seed,resources,list_n_ep,list_n_ep,max_iter)
        analyzer.graph_cost_analysis()
        # analyzer.graph_accuracy_analysis('test')
        # analyzer.graph_accuracy_analysis('train')
        # analyzer.graph_train_vs_test_estimates()

        # Ajuste de parametros
        tuner=CriteriaTuner(algo,env,seed,resources,list_n_ep,list_freq,max_iter)
        opt_n_ep_train,opt_n_ep_test,opt_freq_test=tuner.graph_best_val_tuning()

        # Comparacion de mejor version de criterios existentes
        # grapher=EvolutionGrapher(algo,env,seed,resources,max_iter,list_n_traj_ep=[opt_n_ep_train],list_n_val_ep=[opt_n_ep_test],list_n_val_freq=[opt_freq_test])
        # grapher.graph_rank_evolution()
        # grapher.graph_rank_evolution(True)
        # grapher.graph_magnitude_evolution()
        # grapher.graph_magnitude_evolution(True,True)

class ProcessIndependentAnalysis():
    def __init__(self,process_ids,title,
                 global_deg_metric,local_deg_metric,customized='all',
                 grid_train_n_ep=[],grid_test_n_ep=[],grid_test_freq=[]):# Por si se quieren añadir valores adicionales a los considerados por el tuner
        
        # Guardar las estimaciones de expected episodic reward necesarias en caso de que ya no esten guardadas 
        for process_id in tqdm(process_ids):
            algo,env,seed=Converter.process_id_splitter(process_id)

            for n_ep in grid_train_n_ep:
                Estimator.compute_estimates(algo,env,seed,'16cpu1gpu_mejorado',n_ep,'train')
            for n_ep in grid_test_n_ep:
                Estimator.compute_estimates(algo,env,seed,'16cpu1gpu_mejorado',n_ep,'test')

        # Generar datos de degradacion por proceso y magnitudes por criterio
        analyzer=ProcessIndependentAnalyzer(306,
                                            global_deg_metric=global_deg_metric,local_deg_metric=local_deg_metric,
                                            all_possible_conf=True,
                                            grid_train_n_ep=list(set([500,250,100,50,25,5]+grid_train_n_ep)),
                                            grid_test_n_ep=list(set([500,250,100,50,25,5]+grid_test_n_ep)),
                                            grid_test_freq=list(set([100,50,25,10,5,1]+grid_test_freq)))
        
        # Comparacion de criterios por tiempo
        if customized in ['criteria_by_time']:
            analyzer.graph_best_criteria_by_time(process_ids,[500,250,100,50,25],[150,50,25,5],[5,1],global_deg_metric,local_deg_metric)

        # Analisis avanzado: comparacion de mejores verisones de los tres criterios por nivel de degradacion
        # - Configuraciones optimas para los criterios train y test en las secuencias de cada proceso generador (algo,env,seed)
        # - Version original de test con penalizacion de tiempo extra (la magnitud es la diferencia del truth de la piltica seleccionada con el truth best real en la secuencia que hubiesemos generado invirtiendo todo el tiempo en train)
        # - Para cada nivel de degradacion se consideran las secuencias de cualquir tamaño con esa degradacion
        if customized in ['criteria_by_deg','all']:
            analyzer.graph_best_criteria_by_degradation(process_ids,title,global_deg_metric,local_deg_metric)

        # Analisis de sensibilidad para las configuraciones de los criterios train y test: considerar como criterios las mejores versiones de train y test
        if customized in ['sensitivity_by_deg','all']:
            analyzer.graph_best_train_test_criteria_by_degradation(process_ids,title,'train',global_deg_metric,local_deg_metric)
            analyzer.graph_best_train_test_criteria_by_degradation(process_ids,title,'test',global_deg_metric,local_deg_metric)

        # Analisis para ganar intuicion: 
        # - Configuraciones explicitamente indicadas
        # - Incremento de de dificultad test: sin penalizacion, y con penalizacion disminuyendo frecuencia
        # - Diferentes tamaños de secuencias por separado
        if customized in ['intuition','all']:
            analyzer.graph_gain_intuition_best_criteria_by_degradation(process_ids,[500,250,100,50,25,5],[100,50,25,10,5,1],[0.25,0.5,0.75,1],
                                                                       global_deg_metric,local_deg_metric)

###################################################################################################
experiments_intuition=False
MAEB_ISG_experiments=False

#==================================================================================================
# Main: experimentos realizados para ganar intuicion 
# (esta experimentación esta resumida cronologicamente en experiments_intuition/README/main.pdf )
#==================================================================================================

if experiments_intuition:
    # Analisis de procesos individuales (306 iteraciones equivalen aproximadamente a 10000000 steps de aprendizaje con 16 environments en paralelo)
    for i in [1,2,3,4,5,6,7,8,9,10]:
        SingleProcessAnalisys('PPO','Ant',i,'16cpu1gpu_mejorado',306)
        SingleProcessAnalisys('PPO','Humanoid',i,'16cpu1gpu_mejorado',306)
        SingleProcessAnalisys('PPO','HumanoidStandup',i,'16cpu1gpu_mejorado',306)

    # Analisis general (independiente de proceso): comparacion de criterios por nivel de degradacion (usando diferentes metricas de degradacion)
    all_process_ids=['PPO_Ant_seed1','PPO_Ant_seed2','PPO_Ant_seed3','PPO_Ant_seed4',
                    'PPO_Humanoid_seed1','PPO_Humanoid_seed2','PPO_Humanoid_seed3','PPO_Humanoid_seed4',
                    'PPO_HumanoidStandup_seed1']
    global_deg_metric='mean_update_deg'
    local_deg_metric='greater_prob'
    ProcessIndependentAnalysis(all_process_ids[0:4],'all_Ant',global_deg_metric,local_deg_metric)
    ProcessIndependentAnalysis(all_process_ids[4:8],'all_Humanoid',global_deg_metric,local_deg_metric)
    ProcessIndependentAnalysis(all_process_ids,'all_Ant_Humanoid_HumanoidStandup',global_deg_metric,local_deg_metric)

    global_deg_metric='weighted_mean_best_later_deg'
    local_deg_metric='paired_diff_probpos_meanpos'
    ProcessIndependentAnalysis(all_process_ids[0:4],'all_Ant',global_deg_metric,local_deg_metric)
    ProcessIndependentAnalysis(all_process_ids[4:8],'all_Humanoid',global_deg_metric,local_deg_metric)
    ProcessIndependentAnalysis(all_process_ids,'all_Ant_Humanoid_HumanoidStandup',global_deg_metric,local_deg_metric)

    global_deg_metric='weighted_mean_best_later_deg'
    local_deg_metric='greater_prob'
    ProcessIndependentAnalysis(all_process_ids[0:4],'all_Ant',global_deg_metric,local_deg_metric)
    ProcessIndependentAnalysis(all_process_ids[4:8],'all_Humanoid',global_deg_metric,local_deg_metric)
    ProcessIndependentAnalysis(all_process_ids,'all_Ant_Humanoid_HumanoidStandup',global_deg_metric,local_deg_metric)

    # Analisis general (independiente de proceso): comparacion de criterios por tiempos de aprendizaje
    df1=pd.read_csv('experiments_intuition/results/CriteriaComparison/data/test_affordable_n_ep_by_process.csv')
    df2=pd.read_csv('experiments_intuition/results/CriteriaComparison/data/criteria_conf_by_process.csv')
    print('Maximo numero de episodios promedio con el que se consume menos o igual del 25% de tiempo validando: ',df1['n_ep_0.25'].mean())# 151.11111111111111
    print('Numeros de episodio optimos observados para train: ',list(set(df2['train_n_ep'])))
    print('Frecuencias optimas observadas para test: ',list(set(df2['test_freq'])))

    grid_train_n_ep=[500,250,100,50,25] # Numeros de episodios optimos observados para train
    grid_test_n_ep=[150,50,25,5] # Maximo valor considerado aquel que no supera de media el 25% del tiempo total
    grid_test_freq=[5,1] # Frecuencias optimas observadas despues de los tunings


    global_deg_metric='weighted_mean_best_later_deg'
    local_deg_metric='greater_prob'
    ProcessIndependentAnalysis(all_process_ids,'all_Ant_Humanoid_HumanoidStandup',global_deg_metric,local_deg_metric,
                            grid_train_n_ep=grid_train_n_ep,grid_test_n_ep=grid_test_n_ep,grid_test_freq=grid_test_freq,customized='criteria_by_time')

    local_deg_metric='paired_diff_median'
    ProcessIndependentAnalysis(all_process_ids,'all_Ant_Humanoid_HumanoidStandup',global_deg_metric,local_deg_metric,
                            grid_train_n_ep=grid_train_n_ep,grid_test_n_ep=grid_test_n_ep,grid_test_freq=grid_test_freq,customized='criteria_by_time')

    global_deg_metric='best_last_deg'
    local_deg_metric='greater_prob'
    ProcessIndependentAnalysis(all_process_ids,'all_Ant_Humanoid_HumanoidStandup',global_deg_metric,local_deg_metric,
                            grid_train_n_ep=grid_train_n_ep,grid_test_n_ep=grid_test_n_ep,grid_test_freq=grid_test_freq,customized='criteria_by_time')

    local_deg_metric='paired_diff_median'
    ProcessIndependentAnalysis(all_process_ids,'all_Ant_Humanoid_HumanoidStandup',global_deg_metric,local_deg_metric,
                            grid_train_n_ep=grid_train_n_ep,grid_test_n_ep=grid_test_n_ep,grid_test_freq=grid_test_freq,customized='criteria_by_time')


#==================================================================================================
# Generación de graficas adaptadas para MAEB e ISG charla
#==================================================================================================
def MAEB_ISG_generate_graphs():

    # Para el analisis individual usaremos el mismo proceso que en la experimentacion preliminar del proyecto
    algo='PPO'
    env='Ant'
    seed=1
    resources='16cpu1gpu_mejorado'
    max_iter=306
    list_n_ep=[500,250,100,50,25,5]
    list_freq=[100,50,25,10,5,1]

    analyzer=EstimationAnalyzer(algo,env,seed,resources,list_n_ep,list_n_ep,max_iter)
    analyzer.MAEB_graph_cost_analysis()#ISG
    analyzer.MAEB_graph_train_vs_test_estimates()#MAEB+ISG

    tuner=CriteriaTuner(algo,env,seed,resources,[500,100,5],list_freq,max_iter)
    tuner.MAEB_graph_best_val_tuning()# MAEB+ISG

    # MAEB+ISG analisis conjunto y ISG definicion degradacion
    process_ids= ['PPO_Ant_seed'+str(seed) for seed in range(1,11)]+['PPO_Humanoid_seed'+str(seed) for seed in range(1,11)]+['PPO_HumanoidStandup_seed'+str(seed) for seed in range(1,11)]
    title='all_Ant_Humanoid_HumanoidStandup'
    global_deg_metric='best_last_deg'
    local_deg_metric='paired_diff_probpos'
    grid_train_n_ep=[500]
    grid_test_n_ep=[5]
    grid_test_freq=[1]

    analyzer=ProcessIndependentAnalyzer(306,
                                global_deg_metric=global_deg_metric,local_deg_metric=local_deg_metric,
                                all_possible_conf=False,
                                grid_train_n_ep=grid_train_n_ep,#list(set([500,250,100,50,25,5]+grid_train_n_ep)),
                                grid_test_n_ep=grid_test_n_ep,#list(set([500,250,100,50,25,5]+grid_test_n_ep)),
                                grid_test_freq=grid_test_freq)#list(set([100,50,25,10,5,1]+grid_test_freq)))
    
    analyzer.MAEB_graph_best_criteria_by_time(title,process_ids,grid_train_n_ep,grid_test_n_ep,grid_test_freq,global_deg_metric,local_deg_metric)

    # MAEB definicion de degradación (dibujar en la misma curva varias degradaciones, para que se entienda que estamos monitorizando la degradacion con la verdad absoluta)
    grapher=EvolutionGrapher('PPO','Humanoid',6,'16cpu1gpu_mejorado',306)
    grapher.MAEB_graph_degradation_evolution('best_last_deg','paired_diff_probpos',47)

    # Graficas de analisis comparativo extra (con mas detalle)
    ant_process_ids=['PPO_Ant_seed'+str(seed) for seed in range(1,11)]
    humanoid_process_ids=['PPO_Humanoid_seed'+str(seed) for seed in range(1,11)]
    humanidstandup_process_ids=['PPO_HumanoidStandup_seed'+str(seed) for seed in range(1,11)]
    all_process_ids= ['PPO_Ant_seed'+str(seed) for seed in range(1,11)]+['PPO_Humanoid_seed'+str(seed) for seed in range(1,11)]+['PPO_HumanoidStandup_seed'+str(seed) for seed in range(1,11)]

    analyzer.MAEB2_graph_best_criteria_by_time('all_Ant',ant_process_ids,grid_train_n_ep,grid_test_n_ep,grid_test_freq,global_deg_metric,local_deg_metric)
    analyzer.MAEB2_graph_best_criteria_by_time('all_Humanoid',humanoid_process_ids,grid_train_n_ep,grid_test_n_ep,grid_test_freq,global_deg_metric,local_deg_metric)
    analyzer.MAEB2_graph_best_criteria_by_time('all_HumanoidStandup',humanidstandup_process_ids,grid_train_n_ep,grid_test_n_ep,grid_test_freq,global_deg_metric,local_deg_metric)
    analyzer.MAEB2_graph_best_criteria_by_time(title,all_process_ids,grid_train_n_ep,grid_test_n_ep,grid_test_freq,global_deg_metric,local_deg_metric)

if MAEB_ISG_experiments:
    MAEB_ISG_generate_graphs()

#==================================================================================================
# NUEVO ANALISIS
#==================================================================================================
#--------------------------------------------------------------------------------------------------
# Replicar experimentos SingleProcess mas importantes para nuevos entornos
#--------------------------------------------------------------------------------------------------
#------------- Mas environments
for i in range(1,11):
    SingleProcessAnalisys('PPO','HalfCheetah',i,'16cpu1gpu_mejorado',306)
    #SingleProcessAnalisys('PPO','Walker2d',i,'16cpu1gpu_mejorado',306)

#------------- Configuraciones optimas por entorno
df=pd.read_csv('experiments_intuition/results/CriteriaComparison/data/criteria_conf_by_process.csv')
df_opt_conf_per_env=[]
for env in ['HalfCheetah','Ant','Humanoid','HumanoidStandup']:
    # Filas asociadas al env
    df_env=df[df["process_id"].str.contains(env+'_')]
    opt_train_n_ep=max(set(df_env['train_n_ep']), key=list(df_env['train_n_ep']).count)
    print(df_env.value_counts(subset=["test_n_ep", "test_freq"]))
    opt_test_n_ep,opt_test_freq = df_env.value_counts(subset=["test_n_ep", "test_freq"]).idxmax()
    df_opt_conf_per_env.append(['PPO',env,opt_train_n_ep,opt_test_n_ep,opt_test_freq])
# Todos los entornos
opt_train_n_ep=max(set(df['train_n_ep']), key=list(df_env['train_n_ep']).count)
opt_test_n_ep,opt_test_freq = df.value_counts(subset=["test_n_ep", "test_freq"]).idxmax()
df_opt_conf_per_env.append(['PPO','All',opt_train_n_ep,opt_test_n_ep,opt_test_freq])

df_opt_conf_per_env=pd.DataFrame(df_opt_conf_per_env,columns=['algo','env','train_n_ep','test_n_ep','test_freq'])
df_opt_conf_per_env.to_csv("experiments_intuition/results/CriteriaComparison/data/opt_criteria_conf_by_env.csv", index=False)

#---------------- Comparacion de criterios por entorno
global_deg_metric='relative_worsening_to_improvement'
local_deg_metric='relative_reward_diff'

analyzer=ProcessIndependentAnalyzer(306,grid_test_freq=[1],grid_test_n_ep=[500,100,5],grid_train_n_ep=[500,250,100,50,25,5],
                                    global_deg_metric=global_deg_metric,local_deg_metric=local_deg_metric)

halfcheetah_process_ids=['PPO_HalfCheetah_seed'+str(seed) for seed in range(1,11)]
ant_process_ids=['PPO_Ant_seed'+str(seed) for seed in range(1,11)]
humanoid_process_ids=['PPO_Humanoid_seed'+str(seed) for seed in range(1,11)]
humanidstandup_process_ids=['PPO_HumanoidStandup_seed'+str(seed) for seed in range(1,11)]
process_ids= ['PPO_Ant_seed'+str(seed) for seed in range(1,11)]+['PPO_Humanoid_seed'+str(seed) for seed in range(1,11)]+['PPO_HumanoidStandup_seed'+str(seed) for seed in range(1,11)]+ ['PPO_HalfCheetah_seed'+str(seed) for seed in range(1,11)]

analyzer.MAEB_graph_best_criteria_by_time('All_envs',process_ids,[500],[5],[1],global_deg_metric,local_deg_metric)
analyzer.MAEB_graph_best_criteria_by_time('HalfCheetah',halfcheetah_process_ids,[500],[5],[1],global_deg_metric,local_deg_metric)
analyzer.MAEB_graph_best_criteria_by_time('Ant',ant_process_ids,[50],[5],[1],global_deg_metric,local_deg_metric)
analyzer.MAEB_graph_best_criteria_by_time('Humanoid',humanoid_process_ids,[500],[100],[1],global_deg_metric,local_deg_metric)
analyzer.MAEB_graph_best_criteria_by_time('HumanoidStandup',humanidstandup_process_ids,[500],[500],[1],global_deg_metric,local_deg_metric)

analyzer.MAEB2_graph_best_criteria_by_time('All_envs',process_ids,[500],[5],[1],global_deg_metric,local_deg_metric)
analyzer.MAEB2_graph_best_criteria_by_time('HalfCheetah',halfcheetah_process_ids,[500],[5],[1],global_deg_metric,local_deg_metric)
analyzer.MAEB2_graph_best_criteria_by_time('Ant',ant_process_ids,[50],[5],[1],global_deg_metric,local_deg_metric)
analyzer.MAEB2_graph_best_criteria_by_time('Humanoid',humanoid_process_ids,[500],[100],[1],global_deg_metric,local_deg_metric)
analyzer.MAEB2_graph_best_criteria_by_time('HumanoidStandup',humanidstandup_process_ids,[500],[500],[1],global_deg_metric,local_deg_metric)

#--------------------------------------------------------------------------------------------------
# Graficas para conclusion final: cuando usar que criterio en funcion del tiempo disponible
#--------------------------------------------------------------------------------------------------
 
analyzer.graph_when_use_test(process_ids,5,1,'All_envs',global_deg_metric=global_deg_metric,local_deg_metric=local_deg_metric)
analyzer.graph_when_use_test(process_ids,500,1,'All_envs_500',global_deg_metric=global_deg_metric,local_deg_metric=local_deg_metric)

analyzer.graph_when_use_test(halfcheetah_process_ids,5,1,'HalfCheetah',global_deg_metric=global_deg_metric,local_deg_metric=local_deg_metric)
analyzer.graph_when_use_test(ant_process_ids,5,1,'Ant',global_deg_metric=global_deg_metric,local_deg_metric=local_deg_metric)
analyzer.graph_when_use_test(humanoid_process_ids,100,1,'Humanoid',global_deg_metric=global_deg_metric,local_deg_metric=local_deg_metric)
analyzer.graph_when_use_test(humanidstandup_process_ids,500,1,'HumanoidStandup',global_deg_metric=global_deg_metric,local_deg_metric=local_deg_metric)

'''
TODO:
- bajar del cluster los datos de Walker2d, y reproducir los mismos experimentos con el
(para tener calculadas las estimaciones, configuraciones optimas, degradaciones asociadas,...)

- fijar configuracion optima de train y test global para todos los entornos. Añadir al analisis 
actual (solo max magnitudes) el porcentage de tiempo de validacion con respecto a aprender. La 
configuracion optima de test puede ser el maximo n_ep que no supera el 25%.

- hacer analisis comparativos individuales por entornos (la metrica de magnitud y degradacion
pueden no ser comparables entre entornos).

- en comparaciones pares entre criterios, contar los empates como 0.5 para cada uno en ved de 1.

- dejar mismo tiempo fijo para diferentes entornos (no meternos en convergencia)

- en el analisis que concluye las consecuencias de aplicar cada criterio last-test, en lugar de
la aplicacion simetrica de test, considerar la diagonal inversa a la hecha, empezar en iter. x y
validar el reto de iteraciones hasta el final.

- entender porque es mas costosa la validacion en entornos mas simples (puede que sea la 
longitud de los episodios) y train no es tan malo en entornos simples (puede ser por lo 
tolerante que es el entorno a politicas diferentes)

- modificaciones en definicion de degradacion para mas adelante. Definirla en funcion de f 
es mas simple y concuerda con la definicion del problema, pero causa problemas en comparar
degradaciones de diferentes entornos (por la escala del reward). Definirla en funcion de la
distribucion de reward sobre los estados iniciales se aleja mas de la definicion del problema
y es mas compleja, pero es comparable en diferentes entornos. Hay un paper que habla sobre como
normalizar los rewards de diferentes entornos para que sean comparables!!


FIXME: 
- Si recalculo los truth de la misma manera (los guardados para Ant, Humanoid, HumanoidSatndup),
  no salen igual que la vez pasada (puede ser por la version de la libreria que descomprime).
'''

