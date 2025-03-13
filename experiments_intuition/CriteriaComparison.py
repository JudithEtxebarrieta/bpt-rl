'''
En este scrip se aborda la comparacion grafica de los criterios existentes.

Se consideran 4 procesos de aprendizaje: 

- seed=1,3
algo= PPO, env=Ant, learning time=10000000 steps, 16 CPU para interaccion train y validacion en 500 episodios (y device='auto')

- seed=1,2
algo= PPO, env=Humanoid, learning time=10000000 steps, 16 CPU para interaccion train y validacion en 500 episodios (y device='cpu')

De los 500 datos de episodic reward almacenados:
- 250 para ground truth
- 250 para simular diferentes tamaños de episodios de validacion

Las graficas representan:
- Proceso de aprendizaje con learning-curves (para conocer el nivel de degradacion)
- Criterios con evolucion de rank y magnitud.
- Coste y precision de estimaciones.

TODO: comparar la eficacia de los criterios dependiendo del nivel de paralelizacion usado para aprender y validar.
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
    
    def read_create_estimates_csv(path_csv,seq_size):
        '''
        Crea (si no existe) o lee (si existe) una base de datos para almacenar estimaciones de expected episodic reward por politica de la secuencia.
        '''
        if not os.path.exists(path_csv):
            df_estimates = pd.DataFrame({"n_policy": list(range(seq_size))})
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
        
        # Calcular estimaciones que usaremos como groun truth con los primeros 250 episodios de validacion
        df_test['ep_rewards']=[ Converter.compress_decompress_list(i,compress=False) for i in df_test['ep_rewards']]
        df_val_estimates=Estimator.read_create_estimates_csv(current_path+'/df_val_estimates.csv',df_train.shape[0])
        if 'truth' not in df_val_estimates.columns.tolist():
            df_val_estimates['truth']=[ np.mean(i[:250]) for i in df_test['ep_rewards'] ]
            df_val_estimates.to_csv(current_path+'/df_val_estimates.csv', index=False)
        if 'truth_norm' not in df_val_estimates.columns.tolist():
            df_val_estimates['truth_norm']=[ (i-min(df_val_estimates['truth']))/(max(df_val_estimates['truth'])-min(df_val_estimates['truth'])) for i in df_val_estimates['truth'] ]
            df_val_estimates.to_csv(current_path+'/df_val_estimates.csv', index=False)

        # Calcular estimaciones a partir de datos de train
        if train_test_estimate=='train':
            # Añadir columnas de interes en df_traj_estimates (siempre que ya no esten calculadas previamente)
            df_train['traj_rewards']=[ Converter.compress_decompress_list(i,compress=False) for i in df_train['traj_rewards']]
            df_train['traj_ep_end']=[ Converter.compress_decompress_list(i,compress=False) for i in df_train['traj_ep_end']]
            rollout=np.array(df_train['traj_rewards'][0]).shape[1]

            df_traj_estimates=Estimator.read_create_estimates_csv(current_path+'/df_traj_estimates.csv',df_train.shape[0])
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
                estimates=[np.mean(i[250:(250+n_ep)]) for i in df_test['ep_rewards']]
                val_times=[]
                for i in range(df_test.shape[0]):
                    df_test_elapsed_val_time=Converter.compress_decompress_list(df_test['elapsed_val_time'][i],compress=False)
                    df_test_n_val_ep=Converter.compress_decompress_list(df_test['n_val_ep'][i],compress=False)
                    val_time_until_truth=df_test_elapsed_val_time[df_test_n_val_ep.index(250)] 
                    val_time=df_test_elapsed_val_time[df_test_n_val_ep.index(250+n_ep)] 
                    val_times.append(val_time-val_time_until_truth)
                          
                df_val_estimates[str(n_ep)+'_val_ep']=[Converter.compress_decompress_list(i) for i in zip(estimates,val_times)]
                df_val_estimates.to_csv(current_path+'/df_val_estimates.csv', index=False)
        

    def time_discretizer(algo,env,seed,resources,iter_freq,iter_max):
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
        return list(np.arange(df_train['time_seconds'][0],iter_mean_time*iter_max,iter_mean_time*iter_freq))


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

    def normalize_list(lst):
        lst = np.array(lst)
        return (lst - np.min(lst)) / (np.max(lst) - np.min(lst))
     
class EvolutionGenerator:
    def __init__(self,algo,env,seed,resources):
          
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
    #----------------------------------------------------------------------------------------------
    # Metricas para medir la calidad de la politica seleccionada
    #----------------------------------------------------------------------------------------------
    def real_ranking_position(self,elapsed_time,n_policy):

        last_policy=self.last_policy(elapsed_time,time_in='seconds')
        EER_list=self.df_test_estimates[self.df_test['n_policy']<=last_policy]['truth'].tolist()

        return np.argsort(EER_list)[::-1].tolist().index(n_policy)+1

    def magnitude(self,elapsed_time,n_policy,normalized):

        last_policy=self.last_policy(elapsed_time,time_in='seconds')
        if not normalized:
            EER_list=self.df_test_estimates[self.df_test['n_policy']<=last_policy]['truth'].tolist()
        if normalized:
            EER_list=self.df_test_estimates[self.df_test['n_policy']<=last_policy]['truth_norm'].tolist()


        EER_real_best=max(EER_list)
        EER_criteria_best=EER_list[n_policy]

        return abs(EER_real_best-EER_criteria_best)
    #----------------------------------------------------------------------------------------------
    # Criterios de seleccion
    #----------------------------------------------------------------------------------------------
    def truth_best_policy(self,elapsed_time,time_in='seconds'):
        if time_in=='seconds':
            policy_id=self.df_test_estimates[self.df_train['time_seconds']<=elapsed_time]['truth'].idxmax()
        if time_in=='steps':
            policy_id=self.df_test_estimates[self.df_train['n_timesteps']<=elapsed_time]['truth'].idxmax()

        return policy_id

    def worst_policy(self,elapsed_time,time_in='seconds'):
        if time_in=='seconds':
            policy_id=self.df_test_estimates[self.df_train['time_seconds']<=elapsed_time]['truth'].idxmin()
        if time_in=='steps':
            policy_id=self.df_test_estimates[self.df_train['n_timesteps']<=elapsed_time]['truth'].idxmin()

        return policy_id

    def last_policy(self,elapsed_time,time_in='seconds'):
        '''
        Dado el tiempo transcurrido de aprendizaje, devuelve el indice de la ultima politica visitada.
        '''

        if time_in=='seconds':
            last_policy=self.df_train[self.df_train['time_seconds']<=elapsed_time]['n_policy'].max()
        if time_in=='steps':
            last_policy=self.df_train[self.df_train['n_timesteps']<=elapsed_time]['n_policy'].max()

        return last_policy

    def best_policy_training(self,elapsed_time,n_traj_ep,time_in='seconds'):
        # Lista de EER estimados con datos de train de la secunencia de politicas visitada hasta el momento
        if time_in=='seconds':
            estimated_EER_seq=self.df_train_estimates[self.df_train['time_seconds']<=elapsed_time][str(n_traj_ep)+'_traj_ep'].tolist()
        if time_in=='steps':
            estimated_EER_seq=self.df_train_estimates[self.df_train['n_timesteps']<=elapsed_time][str(n_traj_ep)+'_traj_ep'].tolist()

        # Indice de la politica con mayor mean ER en train
        return estimated_EER_seq.index(max(estimated_EER_seq))

    def best_policy_validation(self,elapsed_time,n_val_ep,freq,time_in='seconds'):

        # Tiempos de validacion con frecuencia constante indicada
        current_val_times=[i for i in freq if i<=elapsed_time]

        # Indices de las politicas asociadas a esos tiempos, sus estimaciones de EER y el tiempo adicional consumido para su calculo
        current_val_policies=[]
        esti_time_seq=[]
        for time in current_val_times:

            if time_in=='seconds':
                policy_id=self.df_train.loc[self.df_train['time_seconds']<=time].index.max()
            if time_in=='steps':
                policy_id=self.df_train.loc[self.df_train['time_seconds']<=time].index.max()

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
                policy_id=self.truth_best_policy(time,time_in='seconds')
            if criteria=='worst':
                policy_id=self.worst_policy(time,time_in='seconds')
            if criteria=='last':
                policy_id=self.last_policy(time,time_in='seconds')
            if criteria=='best_train':
                policy_id=self.best_policy_training(time,n_ep,time_in='seconds')
            if criteria=='best_val':
                policy_id,val_time=self.best_policy_validation(time,n_ep,freq,time_in='seconds')
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
                policy_id=self.truth_best_policy(time,time_in='seconds')
            if criteria=='worst':
                policy_id=self.worst_policy(time,time_in='seconds')
            if criteria=='last':
                policy_id=self.last_policy(time,time_in='seconds')
            if criteria=='best_train':
                policy_id=self.best_policy_training(time,n_ep,time_in='seconds')
            if criteria=='best_val':
                policy_id,val_time=self.best_policy_validation(time,n_ep,freq,time_in='seconds')
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

    def __init__(self,algo,env,seed,resources,
                 list_n_traj_ep,
                 list_n_val_ep, list_n_val_freq,
                 iter_max):
        
        # Primero generar los datos necesarios para las graficas
        for n_ep in tqdm(list_n_traj_ep):
            Estimator.compute_estimates(algo,env,seed,resources,n_ep,'train')
        for n_ep in tqdm(list_n_val_ep):
            Estimator.compute_estimates(algo,env,seed,resources,n_ep,'test')

        # Guardar en una variable las 4 bases de datos
        self.generator=EvolutionGenerator(algo,env,seed,resources)

        # Guardar el resto de variables
        self.algo=algo
        self.env=env
        self.seed=seed
        self.resources=resources
        self.list_n_traj_ep=list_n_traj_ep
        self.list_n_val_ep=list_n_val_ep
        self.list_n_val_freq=list_n_val_freq
        self.iter_max=iter_max

        
    def graph_rank_evolution(self):
        '''
        Dibuja en la misma grafica la evolucion de la posicion en el ranking real para cada posible criterio
        definido a partir de list_n_traj_ep, list_n_val_ep y list_n_val_freq, junto a las curvas truth, worst y last.
        '''

        fig=plt.figure(figsize=[7,5])
        plt.subplots_adjust(left=0.14,bottom=0.4,right=0.86,top=0.92,wspace=0.39,hspace=0.2)
        ax=plt.subplot(111)
        ax.grid(True, which='both',linestyle='--', linewidth=0.8,alpha=0.2)        

        x_times=Estimator.time_discretizer(self.algo,self.env,self.seed,self.resources,1,self.iter_max)

        # Dibujar la curva ideal
        y_trurh_best=self.generator.rank_evolution(x_times,criteria='truth_best') 
        plt.plot(x_times, y_trurh_best, linewidth=1,label='Ground truth')

        # Dibujar la peor curva
        y_worst=self.generator.rank_evolution(x_times,criteria='worst') 
        plt.plot(x_times, y_worst, linewidth=1,label='Worst')

        # Dibujar la curva de criterio last
        y_last=self.generator.rank_evolution(x_times,criteria='last') 
        plt.plot(x_times, y_last, linewidth=1,label='Last')

        # Dibujar las curvas de criterio train
        for n_ep in self.list_n_traj_ep:
            y_best=self.generator.rank_evolution(x_times,n_ep=n_ep,criteria='best_train') 
            plt.plot(x_times, y_best, linewidth=1,label=str(n_ep)+' train ep.')
        
        # Dibujar las curvas de criterio test
        for n_ep in self.list_n_val_ep:
            for freq in self.list_n_val_freq:
                x_times_freq=Estimator.time_discretizer(self.algo,self.env,self.seed,self.resources,freq,self.iter_max)
                y_best,x_extra=self.generator.rank_evolution(x_times,n_ep=n_ep,freq=x_times_freq,criteria='best_val') 
                plt.plot([i+j for i,j in zip(x_times,x_extra)], y_best, linewidth=1,label=str(n_ep)+' val. ep. every '+str(freq)+' policies')


        plt.title('Rank evolution')
        ax.set_xlabel("Total  time")
        ax.set_ylabel("Truth rank of the selected policy")
        plt.legend(title='Criteria',loc="upper center",bbox_to_anchor=(0.5, -0.3),ncol=3)
        plt.savefig('experiments_intuition/results/CriteriaComparison/rank_evolution_'+self.algo+'_'+self.env+str(self.seed)+'_'+self.resources+'.pdf')
        plt.show()
        plt.close()

        
    def graph_magnitude_evolution(self):
        '''
        Dibuja en la misma grafica la evolucion de la magnitud para cada posible criterio
        definido a partir de list_n_traj_ep, list_n_val_ep y list_n_val_freq, junto a las curvas truth, worst y last.
        '''

        fig=plt.figure(figsize=[7,5])
        plt.subplots_adjust(left=0.14,bottom=0.4,right=0.86,top=0.92,wspace=0.39,hspace=0.2)
        ax=plt.subplot(111)
        ax.grid(True, which='both',linestyle='--', linewidth=0.8,alpha=0.2)     

        x_times=Estimator.time_discretizer(self.algo,self.env,self.seed,self.resources,1,self.iter_max)

        # Dibujar la curva ideal
        y_trurh_best=self.generator.magnitude_evolution(x_times,criteria='truth_best') 
        plt.plot(x_times, y_trurh_best, linewidth=1,label='Ground truth')

        # Dibujar la peor curva
        y_worst=self.generator.magnitude_evolution(x_times,criteria='worst') 
        plt.plot(x_times, y_worst, linewidth=1,label='Worst')

        # Dibujar la curva de criterio last
        y_last=self.generator.magnitude_evolution(x_times,criteria='last') 
        plt.plot(x_times, y_last, linewidth=1,label='Last')

        # Dibujar las curvas de criterio train
        for n_ep in self.list_n_traj_ep:
            y_best=self.generator.magnitude_evolution(x_times,n_ep=n_ep,criteria='best_train') 
            plt.plot(x_times, y_best, linewidth=1,label=str(n_ep)+' train ep.')
        
        # Dibujar las curvas de criterio test
        for n_ep in self.list_n_val_ep:
            for freq in self.list_n_val_freq:
                x_times_freq=Estimator.time_discretizer(self.algo,self.env,self.seed,self.resources,freq,self.iter_max)
                y_best,x_extra=self.generator.magnitude_evolution(x_times,n_ep=n_ep,freq=x_times_freq,criteria='best_val') 
                plt.plot([i+j for i,j in zip(x_times,x_extra)], y_best, linewidth=1,label=str(n_ep)+' val. ep. every '+str(freq)+' policies')


        plt.title('Magnitude evolution')
        ax.set_xlabel("Total  time")
        ax.set_ylabel("Difference of truth mean rewards between\n the selected policy and real best")
        plt.legend(title='Criteria',loc="upper center",bbox_to_anchor=(0.5, -0.3),ncol=3)
        plt.savefig('experiments_intuition/results/CriteriaComparison/magnitude_evolution_'+self.algo+'_'+self.env+str(self.seed)+'_'+self.resources+'.pdf')
        plt.show()
        plt.close()

    def graph_magnitude_area(self):
        '''
        Dibuja las areas bajo las curvas de evolucion de magnitud, con la magnitud normalizada.
        El onjetivo de esta grafica es poder distinguir bien cada criterio. El motivo del area es que nos
        interesa minimizar la magnitud para cualquiera que sea el tiempo de aprendizaje.
        '''

        fig=plt.figure(figsize=[4,5])
        plt.subplots_adjust(left=0.14,bottom=0.13,right=0.86,top=0.92,wspace=0.39,hspace=0.2)

        rows=3+len(self.list_n_traj_ep)+len(self.list_n_val_ep)*len(self.list_n_val_freq)
        colors=list(mcolors.TABLEAU_COLORS.keys())
        x_times=Estimator.time_discretizer(self.algo,self.env,self.seed,self.resources,1,self.iter_max)

        ax=plt.subplot(rows,1,1)
        y_trurh_best=self.generator.magnitude_evolution(x_times,criteria='truth_best',normalized=True) 
        plt.plot(x_times, y_trurh_best, linewidth=1,label='Ground truth',color=colors[0])
        plt.fill_between( x_times,y_trurh_best,[0]*len(x_times),alpha=0.5,color=colors[0])
        plt.ylim([-0.1,1.1])
        plt.xticks([])

        plt.title('Normalized magnitude area')

        ax=plt.subplot(rows,1,2)
        y_worst=self.generator.magnitude_evolution(x_times,criteria='worst',normalized=True) 
        plt.plot(x_times, y_worst, linewidth=1,label='Worst',color=colors[1])
        plt.fill_between(x_times,y_worst,[0]*len(x_times),alpha=0.5,color=colors[1])
        plt.ylim([-0.1,1.1])
        plt.xticks([])

        ax=plt.subplot(rows,1,3)
        y_last=self.generator.magnitude_evolution(x_times,criteria='last',normalized=True) 
        plt.plot(x_times, y_last, linewidth=1,label='Last',color=colors[2])
        plt.fill_between(x_times,y_last,[0]*len(x_times),alpha=0.5,color=colors[2])
        plt.ylim([-0.1,1.1])
        plt.xticks([])

        i=4
        for n_ep in self.list_n_traj_ep:
            ax=plt.subplot(rows,1,i)
            y_best=self.generator.magnitude_evolution(x_times,n_ep=n_ep,criteria='best_train',normalized=True) 
            plt.plot(x_times, y_best, linewidth=1,label=str(n_ep)+' train ep.',color=colors[i-1])
            plt.fill_between(x_times,y_best,[0]*len(x_times),alpha=0.5,color=colors[i-1])
            plt.ylim([-0.1,1.1])
            plt.xticks([])
            i+=1

        for n_ep in self.list_n_val_ep:
            for freq in self.list_n_val_freq:
                ax=plt.subplot(rows,1,i)
                x_times_freq=Estimator.time_discretizer(self.algo,self.env,self.seed,self.resources,freq,self.iter_max)
                y_best,x_extra=self.generator.magnitude_evolution(x_times,n_ep=n_ep,freq=x_times_freq,criteria='best_val',normalized=True) 
                plt.plot([i+j for i,j in zip(x_times,x_extra)], y_best, linewidth=1,label=str(n_ep)+' val. ep. every '+str(freq)+' policies',color=colors[i-1])
                plt.fill_between([i+j for i,j in zip(x_times,x_extra)],y_best,[0]*len(x_times),alpha=0.5,color=colors[i-1])
                plt.ylim([-0.1,1.1])
                if i<rows:
                    plt.xticks([])

                i+=1

        ax.set_xlabel("Total  time")
        #ax.set_ylabel("Normaliced magnitude")
        #plt.legend(title='Criteria',loc="upper center",bbox_to_anchor=(0.5, -0.3),ncol=3)
        plt.savefig('experiments_intuition/results/CriteriaComparison/magnitude_area_'+self.algo+'_'+self.env+str(self.seed)+'_'+self.resources+'.pdf')
        plt.show()
        plt.close()

    def learning_curve(self):
        '''
        Evolucion del proceso de apendizaje usando el mean episodic reward que representa el groun truth.
        '''

        fig=plt.figure(figsize=[15,2.5])
        plt.subplots_adjust(left=0.1,bottom=0.2,right=0.94,top=0.82,wspace=0.39,hspace=0.2)
        ax=plt.subplot(111)
        ax.grid(True, which='both',linestyle='--', linewidth=0.8,alpha=0.2)

        # Dibujar curva sin test durante el proceso (ultima politica observada)
        y=[]
        x_policy=[]
        x_times=Estimator.time_discretizer(self.algo,self.env,self.seed,self.resources,1,self.iter_max)

        for time in x_times:
                last_policy=self.generator.df_train[self.generator.df_train['time_seconds']<=time]['n_policy'].max()+1
                y.append(float(self.generator.df_test_estimates[self.generator.df_test_estimates['n_policy']==last_policy]['truth']))
                x_policy.append(last_policy)
        plt.plot(x_times, y, linewidth=1,color='black')

        plt.title('Learning-curve')
        ax.set_xlabel("Total iterations")
        ax.set_ylabel("Truth expected episodic reward\nof the last policy")
        plt.savefig('experiments_intuition/results/CriteriaComparison/learning_curve_'+self.algo+'_'+self.env+str(self.seed)+'_'+self.resources+'.pdf')
        plt.show()
        plt.close()

class EstimationAnalyzer():    

    def __init__(self,algo,env,seed,resources,
                 list_n_traj_ep,
                 list_n_val_ep, list_n_val_freq,
                 iter_max):
        
        # Primero generar los datos necesarios para las graficas
        for n_ep in tqdm(list_n_traj_ep):
            Estimator.compute_estimates(algo,env,seed,resources,n_ep,'train')
        for n_ep in tqdm(list_n_val_ep):
            Estimator.compute_estimates(algo,env,seed,resources,n_ep,'test')

        # Guardar en una variable las 4 bases de datos
        self.generator=EvolutionGenerator(algo,env,seed,resources)

        # Guardar el resto de variables
        self.algo=algo
        self.env=env
        self.seed=seed
        self.resources=resources
        self.list_n_traj_ep=list_n_traj_ep
        self.list_n_val_ep=list_n_val_ep
        self.list_n_val_freq=list_n_val_freq
        self.iter_max=iter_max 


    def matrix_for_cost_analysis(self):

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
        variability= Converter.normalize_list([np.var(Converter.compress_decompress_list(i,compress=False)[:250]) for i in self.generator.df_test['ep_rewards'][:self.iter_max]])
        durability=Converter.normalize_list([np.mean(Converter.compress_decompress_list(i,compress=False)[:250]) for i in self.generator.df_test['ep_lens'][:self.iter_max]])
        matrix=np.vstack((np.array([goodness,variability,durability]),np.array(matrix)))

        return matrix
    
    def matrix_for_accuracy_analysis(self,freq=None,train_or_test='train'):
        matrix=[]

        # Lista de varianzas de los truth en la secuencia catual
        vars=[]
        for i in range(1,self.iter_max+1):
            vars.append(np.var(sorted(self.generator.df_test_estimates['truth'][:i],reverse=True)[:i]))
        matrix.append(Converter.normalize_list(vars))

        # Listas de normalized magnitude para cada posible n_ep
        if train_or_test=='train':
            list_n_ep=self.list_n_traj_ep
            criteria='best_train'
        if train_or_test=='test':
            criteria='best_val'
            list_n_ep=self.list_n_val_ep
            freq=Estimator.time_discretizer(self.algo,self.env,self.seed,self.resources,freq,self.iter_max)

        for n_ep in list_n_ep:
            if train_or_test=='train':
                magnitude=self.generator.magnitude_evolution(self.generator.df_train['time_seconds'][:self.iter_max],n_ep,freq,criteria,normalized=True,for_analyzer=True)
            if train_or_test=='test':
                magnitude,extra_time=self.generator.magnitude_evolution(self.generator.df_train['time_seconds'][:self.iter_max],n_ep,freq,criteria,normalized=True,for_analyzer=True)
            matrix.append(magnitude)

        return np.array(matrix)
    
    def generate_colormap(self,value, cmap_name, vmin, vmax):
        """Asigna un color segun el valor y el mapa de colores indicado"""
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap(cmap_name)
        return cmap(norm(value))

    def colored_matrix_for_cost_analysis(self,matrix,cost_perc_threshold=0.25):
        """Crea una imagen coloreada basada en la matriz para el analisis de los costes de estimacion"""
        rows, cols = matrix.shape
        colored_matrix = np.zeros((rows, cols, 3))  # Matriz para colores RGB
        
        # Filas iniciales (escala azul): goodness, variability y durability de la politica
        for i in range(3):
            for j in range(cols):
                blue_value = matrix[i, j]  # Valor entre 0 y 1
                colored_matrix[i, j] = self.generate_colormap(blue_value, 'Blues', 0, 1)[:3]  # Usar azul para estas filas
        
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
                    colored_matrix[i, j] = self.generate_colormap(cost_perc_threshold - value, 'Greens', 0, cost_perc_threshold)[:3]
                else:
                    colored_matrix[i, j] = self.generate_colormap(value, 'Reds', cost_perc_threshold, np.max(matrix[i]))[:3]
        
        return colored_matrix
    
    def colored_matrix_for_accuracy_analysis(self,matrix):
        rows, cols = matrix.shape
        colored_matrix = np.zeros((rows, cols, 3))  # Matriz para colores RGB
        
        # Fila inicial (escala azul): varianza del truth entre el mejor 25% de las politicas de la secunecia
        for j in range(cols):
            blue_value = matrix[0, j]  # Valor entre 0 y 1
            colored_matrix[0, j] = self.generate_colormap(blue_value, 'Blues_r', 0, 1)[:3]  # Usar azul para estas filas
        
        # Resto de filas (escala de grises): magnitud normalizada
        for i in range(1,rows):
            for j in range(cols):
                gray_value = 1 - matrix[i, j]  # Invertir el valor para la escala de grises
                colored_matrix[i, j] = [gray_value, gray_value, gray_value]
        
        return colored_matrix
    
    def graph_cost_analysis(self):
        # Genara la matriz numerica a partir de los datos
        matrix=self.matrix_for_cost_analysis()

        # Generar matriz de colores
        colored_matrix = self.colored_matrix_for_cost_analysis(matrix)

        fig, ax = plt.subplots(figsize=(20, 6))
        plt.subplots_adjust(left=0.09, bottom=0.2, right=0.81, top=0.82, wspace=0.39, hspace=0.2)
        im = ax.imshow(colored_matrix, aspect='auto')

        # Crear barra de color para la escala de azules
        cbar_ax0 = fig.add_axes([0.82, 0.15, 0.015, 0.7])
        sm0 = plt.cm.ScalarMappable(cmap='Blues', norm=mcolors.Normalize(vmin=0, vmax=1))  # Usar 'Blues' para las filas azules
        sm0.set_array([])
        cbar0 = plt.colorbar(sm0, cax=cbar_ax0)
        cbar0.set_label('Policy characteristic level',fontsize=12)  # Título de la barra de color
        cbar0.ax.yaxis.set_tick_params(rotation=90)
        
        # Crear barra de color para la escala de grises invertida
        cbar_ax0 = fig.add_axes([0.87, 0.15, 0.015, 0.7])
        sm0 = plt.cm.ScalarMappable(cmap='gray_r', norm=mcolors.Normalize(vmin=0, vmax=1))
        sm0.set_array([])
        cbar0 = plt.colorbar(sm0, cax=cbar_ax0)
        cbar0.set_label('Normalized train data',fontsize=12)
        cbar0.ax.yaxis.set_tick_params(rotation=90)
        
        # Crear barra de color para la escala verde invertida
        cbar_ax1 = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        sm1 = plt.cm.ScalarMappable(cmap='Greens_r', norm=mcolors.Normalize(vmin=0, vmax=0.25))  # Usar 'Greens_r' para verde invertido
        sm1.set_array([])
        cbar1 = plt.colorbar(sm1, cax=cbar_ax1)
        cbar1.ax.yaxis.set_tick_params(rotation=90)
        
        # Crear barra de color para la escala roja
        cbar_ax2 = fig.add_axes([0.95, 0.15, 0.015, 0.7])
        sm2 = plt.cm.ScalarMappable(cmap='Reds', norm=mcolors.Normalize(vmin=0.25, vmax=np.max(matrix)))
        sm2.set_array([])
        cbar2 = plt.colorbar(sm2, cax=cbar_ax2)
        cbar2.set_label('Percentage of iteration time',fontsize=12)
        cbar2.ax.yaxis.set_tick_params(rotation=90)
        
        # Etiquetas para los ejes
        row_labels = ['Goodness', 'Variability', 'Durability','Train iter. ep. initis','Train iter. time']+[str(i)+' val. ep.' for i in self.list_n_val_ep]
        ax.set_yticks(np.arange(matrix.shape[0]))  # Establece las posiciones de las filas
        ax.set_yticklabels(row_labels,fontsize=12)  # Establece las etiquetas de las filas
        ax.set_xlabel("Policy in sequence", fontsize=14)
        

        plt.savefig('experiments_intuition/results/CriteriaComparison/estimation_cost_analysis_'+self.algo+'_'+self.env+str(self.seed)+'_'+self.resources+'.pdf')
        plt.show()
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

        # Crear barra de color para la escala de azules
        cbar_ax0 = fig.add_axes([0.82, 0.15, 0.015, 0.7])
        sm0 = plt.cm.ScalarMappable(cmap='Blues_r', norm=mcolors.Normalize(vmin=0, vmax=1))  # Usar 'Blues' para las filas azules
        sm0.set_array([])
        cbar0 = plt.colorbar(sm0, cax=cbar_ax0)
        cbar0.set_label('Normalized episodic reward variance',fontsize=12)  # Título de la barra de color
        cbar0.ax.yaxis.set_tick_params(rotation=90)
        
        # Crear barra de color para la escala de grises invertida
        cbar_ax0 = fig.add_axes([0.87, 0.15, 0.015, 0.7])
        sm0 = plt.cm.ScalarMappable(cmap='gray_r', norm=mcolors.Normalize(vmin=0, vmax=1))
        sm0.set_array([])
        cbar0 = plt.colorbar(sm0, cax=cbar_ax0)
        cbar0.set_label('Normalized magnitude',fontsize=12)
        cbar0.ax.yaxis.set_tick_params(rotation=90)
        
        # Etiquetas para los ejes
        if train_or_test=='train':
            row_labels = ['Difficulty']+[str(i)+' traj. ep.' for i in self.list_n_traj_ep]
        if train_or_test=='test':
            row_labels = ['Difficulty']+[str(i)+' val. ep.' for i in self.list_n_val_ep]
        ax.set_yticks(np.arange(matrix.shape[0]))  # Establece las posiciones de las filas
        ax.set_yticklabels(row_labels,fontsize=12)  # Establece las etiquetas de las filas
        ax.set_xlabel("Policy in sequence", fontsize=14)
        
        plt.savefig('experiments_intuition/results/CriteriaComparison/estimation_accuracy_analysis_'+train_or_test+'_'+self.algo+'_'+self.env+str(self.seed)+'_'+self.resources+'.pdf')
        plt.show()
        plt.close()



        
grapher=EvolutionGrapher('PPO','Ant',1,'16cpu1gpu',[100,50],[100,50],[10,100],306)
analyzer=EstimationAnalyzer('PPO','Ant',1,'16cpu1gpu',[5,25,50,100,250],[5,25,50,100,250],[1,50,100],306)
grapher.learning_curve()
grapher.graph_rank_evolution()
grapher.graph_magnitude_evolution()
grapher.graph_magnitude_area()
analyzer.graph_cost_analysis()
analyzer.graph_accuracy_analysis('test')
analyzer.graph_accuracy_analysis('train')

grapher=EvolutionGrapher('PPO','Ant',3,'16cpu1gpu',[100,50],[100,50],[10,100],306)
analyzer=EstimationAnalyzer('PPO','Ant',3,'16cpu1gpu',[5,25,50,100,250],[5,25,50,100,250],[1,50,100],306)
grapher.learning_curve()
grapher.graph_rank_evolution()
grapher.graph_magnitude_evolution()
grapher.graph_magnitude_area()
analyzer.graph_cost_analysis()
analyzer.graph_accuracy_analysis('test')
analyzer.graph_accuracy_analysis('train')

grapher=EvolutionGrapher('PPO','Humanoid',1,'16cpu',[100,50],[100,50],[10,100],306)
analyzer=EstimationAnalyzer('PPO','Humanoid',1,'16cpu',[5,25,50,100,250],[5,25,50,100,250],[1,50,100],306)
grapher.learning_curve()
grapher.graph_rank_evolution()
grapher.graph_magnitude_evolution()
grapher.graph_magnitude_area()
analyzer.graph_cost_analysis()
analyzer.graph_accuracy_analysis('test')
analyzer.graph_accuracy_analysis('train')

grapher=EvolutionGrapher('PPO','Humanoid',2,'16cpu',[100,50],[100,50],[10,100],306)
analyzer=EstimationAnalyzer('PPO','Humanoid',2,'16cpu',[5,25,50,100,250],[5,25,50,100,250],[1,50,100],306)
grapher.learning_curve()
grapher.graph_rank_evolution()
grapher.graph_magnitude_evolution()
grapher.graph_magnitude_area()
analyzer.graph_cost_analysis()
analyzer.graph_accuracy_analysis('test')
analyzer.graph_accuracy_analysis('train')


