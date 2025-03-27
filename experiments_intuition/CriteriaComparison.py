'''
En este scrip se aborda la comparacion grafica de los criterios existentes.

Se consideran 9 procesos de aprendizaje: 

- seed=1,2,3,4
algo= PPO, env=Ant, learning time=10000000 steps, 16 CPU para interaccion train y validacion en 1000 episodios (y device='auto')

- seed=1,2,3,4
algo= PPO, env=Humanoid, learning time=10000000 steps, 16 CPU para interaccion train y validacion en 1000 episodios (y device='auto')

- seed=1
algo= PPO, env=Humanoid, learning time=10000000 steps, 16 CPU para interaccion train y validacion en 1000 episodios (y device='auto')


De los 1000 datos de episodic reward almacenados:
- 500 para ground truth
- 500 para simular diferentes tamaños de episodios de validacion

Las graficas para procesos de aprendizaje indepedientes (definidos por un algoritmo, environment, semilla y tiempo maximo) representan:
- Proceso de aprendizaje con learning-curves, nivel de degradacion y degradacion por actualizacion.
- Ajuste de la configuracion optima para los criterios con parametros.
- Comaparacion de criterios con evolucion de rank y magnitud.
- Coste y precision de estimaciones para la seleccion.

Las graficas para comparacion de criterios segun degradacion (independiente al proceso) representan:
- Comparacion de los 3 criterios (last, best train, best test) en su mejor version por nivel de degradacion.
- Comparacion de criterios train y test para diferentes configuraciones (las registradas como optimas para procesos individuales) por nivel de degradacion. (analisis de sensibilidad)

TODO: comparar la eficacia de los criterios dependiendo del nivel de paralelizacion usado para aprender y validar (mas adelante, ahora 16 CPU con 1 GPU).
- Cuantos menos CPU-> mayor degradacion (?)
- Cuantos menos CPU-> mayor diferencia en validation time vs. interaction time (?)
- Cuantos menos CPU-> peores estimaciones train vs. estimaciones validacion (?)

NOTE: aunque SB3 no este pensado para aprovechar los GPU, he observado que asignandole GPUs va mas rapido.
'''

import os, sys
import pandas as pd
import matplotlib.pyplot as plt
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


parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Estimator:

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
        
        # Calcular estimaciones que usaremos como groun truth con los primeros 500 episodios de validacion
        df_test['ep_rewards']=[ Converter.compress_decompress_list(i,compress=False) for i in df_test['ep_rewards']]
        df_val_estimates=Estimator.read_create_estimates_csv(current_path+'/df_val_estimates.csv',df_train.shape[0])
        if 'truth' not in df_val_estimates.columns.tolist():
            df_val_estimates['truth']=[ np.mean(i[:500]) for i in df_test['ep_rewards'] ]
            df_val_estimates.to_csv(current_path+'/df_val_estimates.csv', index=False)
        if 'truth_norm' not in df_val_estimates.columns.tolist():
            df_val_estimates['truth_norm']=[ (i-min(df_val_estimates['truth']))/(max(df_val_estimates['truth'])-min(df_val_estimates['truth'])) for i in df_val_estimates['truth'] ]
            df_val_estimates.to_csv(current_path+'/df_val_estimates.csv', index=False)
        
        # Tambien estimaciones de degradacion para añadir a las learning-curve mas informativas (empezar a considerar la degradacion despues del 10% del tiempo)
        #if 'update_degradation' not in df_val_estimates.columns.tolist():
        update_degradations,update_dominances=Estimator.compute_update_degradations(algo,env,seed,resources,df_test.shape[0],also_dominance=True)
        df_val_estimates['update_dominances']=[0 for _ in range(int(df_test.shape[0]*.1))]+update_dominances[int(df_test.shape[0]*.1):]
        df_val_estimates['update_degradation']=[0 for _ in range(int(df_test.shape[0]*.1))]+update_degradations[int(df_test.shape[0]*.1):]
        df_val_estimates['degradation_level']=[0 for _ in range(int(df_test.shape[0]*.1))]+[np.mean(df_val_estimates['update_degradation'][int(df_test.shape[0]*.1):i+1]) for i in range(int(df_test.shape[0]*.1),df_test.shape[0])]
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

    def compute_update_degradations(algo,env,seed,resources, iter_max,also_dominance=False):
        current_path=parent_dir+'/_bender/project_SB3/data/'+str(algo)+'_'+str(env)+'_seed'+str(seed)+'_'+resources

        def Cp_estimation(A, B):
            ''' 
            Estimacion de la probabilidad de que las variabes aleatorias de las que provienen las muestras A y B cumplan: X_A < X_B 
            https://www.tandfonline.com/doi/full/10.1080/10618600.2022.2084405
            '''   
            sign_matrix = np.sign(B[:, None] - A)  # Matriz de comparacion
            return (np.sum(sign_matrix) / (2 * len(A)**2)) + 0.5
        
        def update_degradation(X_current,X_prev):
            '''
            Transformacion de dominancia C_p para definir nuestra degradacion por actualizacion (valor en [0,1], 0 no hay degradacion y 
            cuanto mas cerca de 1 mas degradacion). Esta funcion normaliza C_o en [0,1] cuando C_p toma valores en (0.5,1] (i.e. X_prev domina
            a X_current) y 0 en los demas casos (X_current domina a X_prev o son iguales).
            '''
            dominance=Cp_estimation(np.array(X_current),np.array(X_prev))
            indicator=[1 if dominance>0.5 else 0][0]
            return 2*(dominance-0.5)*indicator

        # Leer base de datos test
        df_test=pd.read_csv(current_path+'/df_val.csv')
        df_test['ep_rewards']=[Converter.compress_decompress_list(i,compress=False) for i in df_test['ep_rewards']][:iter_max]

        # Calcular vector de degradaciones por actualizacion
        update_degradations=[0] # En la inicializacion de la secuencia de politicas no hay degradacion
        update_dominances=[0]
        for row in range(1,iter_max):
            X_current=df_test.loc[row,'ep_rewards'][:500] 
            X_prev=df_test.loc[row-1,'ep_rewards'][:500]
            update_degradations.append(update_degradation(X_current,X_prev))
            update_dominances.append(Cp_estimation(np.array(X_current),np.array(X_prev)))

        if also_dominance:
            return update_degradations, update_dominances
        else:
            return update_degradations
class Converter:

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
        min_value=min(value_list)
        value_list.remove(min_value)
        return np.mean([min_value/value if value!=0 else 1 for value in value_list])


class EvolutionGenerator:
    def __init__(self,algo,env,seed,resources,perc_time_start):
          
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
    # Metricas para medir la calidad de la politica seleccionada
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
    
class EvolutionGrapher:

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

    def graph_degradation_evolution(self):

        fig, axes = plt.subplots(4,4,figsize=(12,5),gridspec_kw={'height_ratios': [1,3,1,3]})
        plt.subplots_adjust(left=0.08,bottom=0.25,right=0.8,top=0.92,wspace=0.39,hspace=0.1)
        list_axes=[ax for _,ax in enumerate(axes.flat)]
        for i in range(12):
            list_axes[i].set_visible(False)

        def KDE_consecutive_policies(n_policy,ax,title,update_dom,update_deg,truth_deg):

            # Obtener los ep. reward truth normalizados de las dos politicas de interes
            column_ep_rewards=[Converter.compress_decompress_list(i,compress=False)[:500] for i in self.generator.df_test['ep_rewards']]
            all_ep_rw=column_ep_rewards[n_policy-1][:500]+column_ep_rewards[n_policy][:500]
            max_ep_rw=max(all_ep_rw)
            min_ep_rw=min(all_ep_rw)
            prev_ER=Converter.normalize_list(column_ep_rewards[n_policy-1][:500],min_ep_rw,max_ep_rw)
            current_ER=Converter.normalize_list(column_ep_rewards[n_policy][:500],min_ep_rw,max_ep_rw)

            # KDE a partir de las dos muestras anteriores
            avg_ep_rw=[]
            x_position=[0]
            x_d=np.linspace(0,1, 1000)
            for sample in [prev_ER,current_ER]:
                x=np.array(sample)
                kde = KernelDensity(bandwidth=0.05, kernel='gaussian')
                kde.fit(x[:, None])
                y_prob = np.exp(kde.score_samples(x_d[:, None]))

                ax.plot(np.full_like(x, -0.01)+x_position[-1],x, '_', markeredgewidth=1,color='grey',alpha=0.5)
                ax.fill_betweenx( x_d,y_prob+x_position[-1],min(y_prob)+x_position[-1],color='red',alpha=0.5)

                x_position.append(max(y_prob)*2)
                avg_ep_rw.append(np.mean(x))

            ax.plot(x_position[:2],avg_ep_rw,color='black')
            ax.set_xlabel(title+'\n $D_i=$'+str(format(update_dom, ".2e"))+'\n$\delta_i=$'+str(format(update_deg, ".2e"))+'\n$\Delta f^-_i=$'+str(format(truth_deg, ".2e")))
            ax.set_xticks(ticks=x_position[:2], labels=[str(n_policy-1),str(n_policy)])

        # Nivel de degradacion
        ax=plt.subplot(4,4,(1,4))
        degradation_level=self.generator.df_test_estimates['degradation_level']
        ax.imshow(np.array(degradation_level).reshape(1, -1), cmap="gray_r", aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Degradation evolution')

        # Learning-curve con lineas verticales que representan los update degradations
        ax=plt.subplot(4,4,(5,8))
        ax.grid(True, which='both',linestyle='--', linewidth=0.8,alpha=0.2)

        cmap_white_to_red = mcolors.LinearSegmentedColormap.from_list("white_red", ["white", "red"])
        for i in range(self.iter_max):
            cmap=Converter.generate_colormap(self.generator.df_test_estimates['update_degradation'][i],cmap_white_to_red,0,1)
            ax.axvline(x=i, color=cmap, linestyle='-', linewidth=1) 

        plt.plot(list(range(self.iter_max))[:int(self.iter_max*.1)], self.generator.df_test_estimates['truth'][:int(self.iter_max*.1)], linewidth=.5,linestyle='--',color='black')
        plt.plot(list(range(self.iter_max))[int(self.iter_max*.1)-1:], self.generator.df_test_estimates['truth'][int(self.iter_max*.1)-1:], linewidth=1,color='black')
        plt.xlim([0,self.iter_max])
        ax.set_ylabel('Episodic reward')
        ax.set_xlabel('Learning iteration ($i$)')

        # KDE de iteracion actual y previa para comparar similitud entre metrica de degradacion y estimacion truth
        update_dominances=np.array(self.generator.df_test_estimates['update_dominances'][int(self.iter_max*0.1):])
        update_degradation=abs(np.array(self.generator.df_test_estimates['update_degradation'][int(self.iter_max*0.1):])) # abs porque al hacer np.array los ceros salen -0.0
        truth_degradation=[0]+[self.generator.df_test_estimates['truth'][i-1]-self.generator.df_test_estimates['truth'][i] for i in range(1,self.iter_max)]
        truth_degradation=abs(Converter.normalize_list([i if i>0 else 0 for i in truth_degradation]))[int(self.iter_max*0.1):]

        true_negative=min((i for i, deg in enumerate(truth_degradation) if deg==0),key=lambda i: update_degradation[i])+int(self.iter_max*0.1)
        true_positive=max((i for i, deg in enumerate(truth_degradation) if deg>0),key=lambda i: update_degradation[i])+int(self.iter_max*0.1)
        false_negative=max((i for i, deg in enumerate(truth_degradation) if deg==0),key=lambda i: update_degradation[i])+int(self.iter_max*0.1)
        false_positive=max((i for i, deg in enumerate(update_degradation) if deg==0),key=lambda i: truth_degradation[i])+int(self.iter_max*0.1)

        ax=plt.subplot(4,4,13) 
        KDE_consecutive_policies(true_negative,ax,'True negative ($i$='+str(true_negative)+')',
                                      update_dominances[true_negative-int(self.iter_max*0.1)],
                                      update_degradation[true_negative-int(self.iter_max*0.1)],
                                      truth_degradation[true_negative-int(self.iter_max*0.1)])
        ax.set_ylabel('Normalized\nepisodic reward')

        ax=plt.subplot(4,4,14)
        KDE_consecutive_policies(true_positive,ax,'True positive ($i$='+str(true_positive)+')',
                                      update_dominances[true_positive-int(self.iter_max*0.1)],
                                      update_degradation[true_positive-int(self.iter_max*0.1)],
                                      truth_degradation[true_positive-int(self.iter_max*0.1)])

        ax=plt.subplot(4,4,15)
        KDE_consecutive_policies(false_negative,ax,'False negative ($i$='+str(false_negative)+')',
                                      update_dominances[false_negative-int(self.iter_max*0.1)],
                                      update_degradation[false_negative-int(self.iter_max*0.1)],
                                      truth_degradation[false_negative-int(self.iter_max*0.1)])
        ax=plt.subplot(4,4,16)
        KDE_consecutive_policies(false_positive,ax,'False positive ($i$='+str(false_positive)+')',
                                      update_dominances[false_positive-int(self.iter_max*0.1)],
                                      update_degradation[false_positive-int(self.iter_max*0.1)],
                                      truth_degradation[false_positive-int(self.iter_max*0.1)])

        # Barras de colores
        Converter.generate_colorbar(fig,[0.84, 0.6, 0.015, 0.3],cmap_white_to_red,[0,max(update_degradation)],'Update degradation ($\delta_i$)')
        Converter.generate_colorbar(fig,[0.92, 0.6, 0.015, 0.3],'Greys',[0,max(degradation_level)],'Degradation level ($\overline{\delta}_i$)')

        plt.savefig('experiments_intuition/results/CriteriaComparison/figures/'+self.algo+'_'+self.env+'_seed'+str(self.seed)+'/improved_learning_curve.pdf')
        #plt.show()
        plt.close()

class EstimationAnalyzer():    

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
        for i in range(2,len(self.list_n_val_ep)+2):
            matrix[i]=np.array(matrix[i])/np.array(matrix[1])

        # Normalizar tiempos de interaccion
        matrix[1]=Converter.normalize_list(matrix[1])

        # Listas para las identificar el tipo de politica (buena/mala, estocastica/determinista, duradera/bolatil)
        goodness=Converter.normalize_list(self.generator.df_test_estimates['truth'][:self.iter_max])
        variability= Converter.normalize_list([np.var(Converter.compress_decompress_list(i,compress=False)[:500]) for i in self.generator.df_test['ep_rewards'][:self.iter_max]])
        durability=Converter.normalize_list([np.mean(Converter.compress_decompress_list(i,compress=False)[:500]) for i in self.generator.df_test['ep_lens'][:self.iter_max]])
        matrix=np.vstack((np.array([goodness,variability,durability]),np.array(matrix)))

        return matrix
    
    def matrix_for_accuracy_analysis(self,freq=None,train_or_test='train'):
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
        matrix.append(self.generator.df_test_estimates['degradation_level'][self.start_iter-1:self.iter_max])

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

        
class CriteriaTuner():

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
        path_csv='experiments_intuition/results/CriteriaComparison/criteria_conf_by_process.csv'
        if not os.path.exists(path_csv):
            df = pd.DataFrame(columns=['process_id','train_n_ep','test_n_ep','test_freq'])
            df.to_csv(path_csv, index=False)
        
        with open(path_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([self.algo+'_'+self.env+'_seed'+str(self.seed)]+opt_conf) 
        Converter.from_csv_to_png('experiments_intuition/results/CriteriaComparison','criteria_conf_by_process')
        return opt_conf
    
class SingleProcessAnalisys():
    def __init__(self,algo,env,seed,resources,max_iter):
            
        # Generar directorio donde se almacenaran las graficas.
        os.makedirs('experiments_intuition/results/CriteriaComparison/figures/'+algo+'_'+env+'_seed'+str(seed), exist_ok=True)
        
        # Valores de parametros a considerara (se hara el tuning entre estos)
        list_n_ep=[500,250,100,50,25,5]# Como las bases de datos "mejoradas" tienen almacenadas 1000 validaciones por politica
        list_freq=[100,50,25,10,5,1]
        
        # Representacion del proceso
        grapher=EvolutionGrapher(algo,env,seed,resources,max_iter)
        grapher.learning_curve()
        grapher.graph_degradation_evolution()

        # Analisis de estimaciones (coste y precision de seleccion)
        analyzer=EstimationAnalyzer(algo,env,seed,resources,list_n_ep,list_n_ep,max_iter)
        analyzer.graph_cost_analysis()
        analyzer.graph_accuracy_analysis('test')
        analyzer.graph_accuracy_analysis('train')
        analyzer.graph_train_vs_test_estimates()

        # Ajuste de parametros
        tuner=CriteriaTuner(algo,env,seed,resources,list_n_ep,list_freq,max_iter)
        opt_n_ep_train,opt_n_ep_test,opt_freq_test=tuner.graph_best_val_tuning()

        # Comparacion de mejor version de criterios existentes
        grapher=EvolutionGrapher(algo,env,seed,resources,max_iter,list_n_traj_ep=[opt_n_ep_train],list_n_val_ep=[opt_n_ep_test],list_n_val_freq=[opt_freq_test])
        grapher.graph_rank_evolution()
        grapher.graph_rank_evolution(True)
        grapher.graph_magnitude_evolution()
        grapher.graph_magnitude_evolution(True,True)

class ProcessIndependentAnalisys():
    def __init__(self,iter_max,perc_time_start=0.1,all_possible_conf=False):

        start_iter=int(iter_max*perc_time_start)-1
        
        # Mirar si ya se ha llamado a esta funcion previamente (si existen las bases de datos necesrias, si no inicializarlas)
        df_degradation=Estimator.read_create_estimates_csv('experiments_intuition/results/CriteriaComparison/data/update_degradation.csv',iter_max,start_iter)
        df_last_mag=Estimator.read_create_estimates_csv('experiments_intuition/results/CriteriaComparison/data/criteria_last_mag.csv',iter_max,start_iter)
        df_train_mag=Estimator.read_create_estimates_csv('experiments_intuition/results/CriteriaComparison/data/criteria_best_train_mag.csv',iter_max,start_iter)
        df_test_mag=Estimator.read_create_estimates_csv('experiments_intuition/results/CriteriaComparison/data/criteria_best_test_mag.csv',iter_max,start_iter)

        # Leer base de datos donde se almacenan las mejores configuraciones de los criterios por proceso (criteria_conf_by_process.csv)
        df_conf=pd.read_csv('experiments_intuition/results/CriteriaComparison/data/criteria_conf_by_process.csv')

        
        # Calcular y guardar por proceso: degradaciones por actualizacion y evolucion de magnitudes por criterio (si no estan ya guardados)
        for row in tqdm(range(df_conf.shape[0])):
            process_id=df_conf.loc[row,'process_id']
            algo,env,seed=Converter.process_id_splitter(process_id)
            
            # Si los datos del proceso no estan ya almacenados calcularlos y almacenarlos
            if process_id not in list(df_degradation.columns):

                # Calcular y guardar las estimaciones de degradacion por actualizacion
                df_degradation[process_id]=Estimator.compute_update_degradations(algo,env,seed,'16cpu1gpu_mejorado',iter_max)[start_iter:]

                # Configuracion optima de los criterios train y test 
                train_n_ep,test_n_ep,test_freq=df_conf.loc[row,['train_n_ep','test_n_ep','test_freq']]

                # Calcular y guardar la evolucion de magnitud de cada criterio (las estimaciones para las configuraciones anteriores las tengo guardadas porque ya he tenido que ejecutar el tuner)
                generator=EvolutionGenerator(algo,env,seed,'16cpu1gpu_mejorado',perc_time_start)
                min_time=generator.df_train.loc[start_iter,'time_seconds']
                x_times=generator.df_train['time_seconds'].tolist()[start_iter:iter_max]  
                x_times_with_freq=Estimator.time_discretizer(algo,env,seed,'16cpu1gpu_mejorado',test_freq,iter_max,min_time)
                df_last_mag[process_id]=generator.magnitude_evolution(x_times,criteria='last',normalized=True)
                df_train_mag[process_id]=generator.magnitude_evolution(x_times,n_ep=train_n_ep,criteria='best_train',normalized=True)
                df_test_mag[process_id]=generator.magnitude_evolution(x_times,n_ep=test_n_ep,freq=x_times_with_freq,criteria='best_val',normalized=True)[0]

            # Almaxenar datos adicionales para posibles combinaciones de configuraciones optimas
            if all_possible_conf:
                all_train_n_ep=list(set(df_conf['train_n_ep']))
                all_test_n_ep_freq=list(set(tuple(pair) for pair in zip(df_conf['test_n_ep'],df_conf['test_freq'])))

                # Calcular y guardar la evolucion de magnitud de cada criterio (las estimaciones para las configuraciones anteriores las tengo guardadas porque ya he tenido que ejecutar el tuner)
                generator=EvolutionGenerator(algo,env,seed,'16cpu1gpu_mejorado',perc_time_start)
                min_time=generator.df_train.loc[start_iter,'time_seconds']
                x_times=generator.df_train['time_seconds'].tolist()[start_iter:iter_max]  

                for n_ep in all_train_n_ep:
                    if process_id+'_'+str(n_ep) not in list(df_train_mag.columns):
                        df_train_mag[process_id+'_'+str(n_ep)]=generator.magnitude_evolution(x_times,n_ep=n_ep,criteria='best_train',normalized=True)

                for n_ep,freq in all_test_n_ep_freq:
                    if process_id+'_'+str(n_ep)+'_'+str(freq) not in list(df_test_mag.columns):
                        x_times_with_freq=Estimator.time_discretizer(algo,env,seed,'16cpu1gpu_mejorado',freq,iter_max,min_time)
                        df_test_mag[process_id+'_'+str(n_ep)+'_'+str(freq)]=generator.magnitude_evolution(x_times,n_ep=n_ep,freq=x_times_with_freq,criteria='best_val',normalized=True)[0]

            # Guardar cambios en bases de datos
            df_degradation.to_csv('experiments_intuition/results/CriteriaComparison/data/update_degradation.csv', index=False)
            df_last_mag.to_csv('experiments_intuition/results/CriteriaComparison/data/criteria_last_mag.csv', index=False)
            df_train_mag.to_csv('experiments_intuition/results/CriteriaComparison/data/criteria_best_train_mag.csv', index=False)
            df_test_mag.to_csv('experiments_intuition/results/CriteriaComparison/data/criteria_best_test_mag.csv', index=False)

        self.df_degradation=df_degradation
        self.df_last_mag=df_last_mag
        self.df_train_mag=df_train_mag
        self.df_test_mag=df_test_mag
        self.iter_max=iter_max

    def read_generate_criteria_rank_by_degradation_data(self,process_ids):
        # Generar datos necesarios para la grafica si ya no estan generados
        path_csv='experiments_intuition/results/CriteriaComparison/criteria_rank_by_degradation.csv'
        if not os.path.exists(path_csv):
            df = pd.DataFrame(columns=['process_id','degradation_level','rank_last','rank_best_train','rank_best_test','not_first_prob'])
            os.makedirs(os.path.dirname(path_csv), exist_ok=True)
            df.to_csv(path_csv, index=False)
        else:
            df=pd.read_csv(path_csv)

        for process_id in process_ids:
            degradation_levels=[np.mean(self.df_degradation[process_id][:i]) for i in range(self.iter_max)]

            criteria_mag=np.array([self.df_last_mag[process_id],self.df_train_mag[process_id],self.df_test_mag[process_id]]).T
            criteria_rankings=[Converter.from_list_to_ranking(magnitudes) for magnitudes in criteria_mag]
            not_first_probs=[Converter.from_list_to_prob_not_first(magnitudes) for magnitudes in criteria_mag]

            rows_to_add=[[process_id,degradation_level,criteria_ranking[0],criteria_ranking[1],criteria_ranking[2],not_first_prob]
                         for degradation_level, criteria_ranking, not_first_prob in zip(degradation_levels,criteria_rankings,not_first_probs)]
            df_new = pd.DataFrame(rows_to_add, columns=df.columns)
            df = pd.concat([df, df_new], ignore_index=True)

        df.to_csv('experiments_intuition/results/CriteriaComparison/data/criteria_rank_by_degradation.csv', index=False)

        return df
    
    def read_generate_train_test_criteria_rank_by_degradation_data(self,process_ids,train_or_test):
        
        
        df_conf=pd.read_csv('experiments_intuition/results/CriteriaComparison/data/criteria_conf_by_process.csv')
        all_train_n_ep=list(set(df_conf['train_n_ep']))
        all_test_n_ep_freq=list(set(tuple(pair) for pair in zip(df_conf['test_n_ep'],df_conf['test_freq'])))
        
        if train_or_test=='train':
            process_id_suffix=[str(n_ep) for n_ep in all_train_n_ep]
        if train_or_test=='test':
            process_id_suffix=[str(n_ep)+'_'+str(freq) for n_ep,freq in all_test_n_ep_freq]
            
        # Generar datos necesarios para la grafica si ya no estan generados
        path_csv='experiments_intuition/results/CriteriaComparison/'+train_or_test+'_criteria_rank_by_degradation.csv'
        if not os.path.exists(path_csv):
            df = pd.DataFrame(columns=['process_id','degradation_level']+['rank_'+i for i in process_id_suffix]+['not_first_prob'])
            os.makedirs(os.path.dirname(path_csv), exist_ok=True)
            df.to_csv(path_csv, index=False)
        else:
            df=pd.read_csv(path_csv)

        for process_id in process_ids:
            degradation_levels=[np.mean(self.df_degradation[process_id][:i]) for i in range(self.iter_max)]

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

        df.to_csv('experiments_intuition/results/CriteriaComparison/data/'+train_or_test+'_criteria_rank_by_degradation.csv', index=False)

        return df


    def matrix_best_criteria_by_degradation(self,process_ids):

        df=self.read_generate_criteria_rank_by_degradation_data(process_ids)
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
    
    def matrix_train_test_criteria_by_degradation(self,process_ids,train_or_test):
        
        df=self.read_generate_train_test_criteria_rank_by_degradation_data(process_ids,train_or_test)
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
            for j in range(3,df_deg_level.shape[1]-1):
                perc_list.append((df_deg_level.iloc[:, j]==1).sum()/df_deg_level.shape[0])
                not_first_prob_list.append(df_deg_level[df_deg_level.iloc[:, j]==1]['not_first_prob'].mean())

            matrix_perc.append(perc_list)
            matrix_prob.append(not_first_prob_list)
            num_data_per_level.append(df_deg_level.shape[0])

        return np.array(matrix_perc),np.array(matrix_prob), degradation_intervals, num_data_per_level, df.columns[3:df_deg_level.shape[1]-1].str.replace('rank_', '', regex=False).tolist()

    def graph_best_criteria_by_degradation(self,process_ids,title):

        # Generar matriz numerica para la grafica
        matrix, matrix_prob,degradation_intervals,num_data_per_level=self.matrix_best_criteria_by_degradation(process_ids)
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

        plt.savefig('experiments_intuition/results/CriteriaComparison/figures/best_criterion_by_degradation_'+title+'.pdf')
        #plt.show()

    def graph_best_train_test_criteria_by_degradation(self,process_ids,title,train_or_test):
        # Generar matriz numerica para la grafica
        matrix, matrix_prob,degradation_intervals,num_data_per_level,labels=self.matrix_train_test_criteria_by_degradation(process_ids,train_or_test)
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

        plt.savefig('experiments_intuition/results/CriteriaComparison/figures/'+train_or_test+'_criterion_by_degradation_'+title+'.pdf')
        #plt.show()
 

# Analisis de procesos individuales (306 iteraciones equivalen aproximadamente a 10000000 steps de aprendizaje con 16 environments en paralelo)
SingleProcessAnalisys('PPO','Ant',1,'16cpu1gpu_mejorado',306)
SingleProcessAnalisys('PPO','Ant',2,'16cpu1gpu_mejorado',306)
SingleProcessAnalisys('PPO','Ant',3,'16cpu1gpu_mejorado',306)
SingleProcessAnalisys('PPO','Ant',4,'16cpu1gpu_mejorado',306)

SingleProcessAnalisys('PPO','Humanoid',1,'16cpu1gpu_mejorado',306)
SingleProcessAnalisys('PPO','Humanoid',2,'16cpu1gpu_mejorado',306)
SingleProcessAnalisys('PPO','Humanoid',3,'16cpu1gpu_mejorado',306)
SingleProcessAnalisys('PPO','Humanoid',4,'16cpu1gpu_mejorado',306)

SingleProcessAnalisys('PPO','HumanoidStandup',1,'16cpu1gpu_mejorado',306)

# Analisis general (independiente de proceso)
analyzer=ProcessIndependentAnalisys(306,all_possible_conf=True)
analyzer.graph_best_criteria_by_degradation(['PPO_Ant_seed1'],'Ant1')
analyzer.graph_best_criteria_by_degradation(['PPO_Ant_seed1','PPO_Ant_seed2','PPO_Ant_seed3','PPO_Ant_seed4'],'all_Ant')
analyzer.graph_best_criteria_by_degradation(['PPO_Humanoid_seed1','PPO_Humanoid_seed2','PPO_Humanoid_seed3','PPO_Humanoid_seed4'],'all_Humanoid')
analyzer.graph_best_criteria_by_degradation(['PPO_HumanoidStandup_seed1'],'HumanoidStandup1')
analyzer.graph_best_criteria_by_degradation(['PPO_Ant_seed1','PPO_Ant_seed2','PPO_Ant_seed3','PPO_Ant_seed4',
                                              'PPO_Humanoid_seed1','PPO_Humanoid_seed2','PPO_Humanoid_seed3','PPO_Humanoid_seed4'],'all_Ant_Humanoid')
analyzer.graph_best_criteria_by_degradation(['PPO_Ant_seed1','PPO_Ant_seed2','PPO_Ant_seed3','PPO_Ant_seed4',
                                              'PPO_Humanoid_seed1','PPO_Humanoid_seed2','PPO_Humanoid_seed3','PPO_Humanoid_seed4',
                                              'PPO_HumanoidStandup_seed1'],'all_Ant_Humanoid_HumanoidStandup')

# Sensibilidad a configuracion
analyzer.graph_best_train_test_criteria_by_degradation(['PPO_Ant_seed1','PPO_Ant_seed2','PPO_Ant_seed3','PPO_Ant_seed4'],'all_Ant','train')
analyzer.graph_best_train_test_criteria_by_degradation(['PPO_Ant_seed1','PPO_Ant_seed2','PPO_Ant_seed3','PPO_Ant_seed4'],'all_Ant','test')
analyzer.graph_best_train_test_criteria_by_degradation(['PPO_Humanoid_seed1','PPO_Humanoid_seed2','PPO_Humanoid_seed3','PPO_Humanoid_seed4'],'all_Humanoid','train')
analyzer.graph_best_train_test_criteria_by_degradation(['PPO_Humanoid_seed1','PPO_Humanoid_seed2','PPO_Humanoid_seed3','PPO_Humanoid_seed4'],'all_Humanoid','test')

analyzer.graph_best_train_test_criteria_by_degradation(['PPO_Ant_seed1','PPO_Ant_seed2','PPO_Ant_seed3','PPO_Ant_seed4',
                                              'PPO_Humanoid_seed1','PPO_Humanoid_seed2','PPO_Humanoid_seed3','PPO_Humanoid_seed4',
                                              'PPO_HumanoidStandup_seed1'],'all_Ant_Humanoid_HumanoidStandup','train')
analyzer.graph_best_train_test_criteria_by_degradation(['PPO_Ant_seed1','PPO_Ant_seed2','PPO_Ant_seed3','PPO_Ant_seed4',
                                              'PPO_Humanoid_seed1','PPO_Humanoid_seed2','PPO_Humanoid_seed3','PPO_Humanoid_seed4',
                                              'PPO_HumanoidStandup_seed1'],'all_Ant_Humanoid_HumanoidStandup','test')