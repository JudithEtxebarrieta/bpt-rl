'''
Este script permite generar en el cluster los df necesarios para el analisis a partir de los df_traj.csv y df_val.csv almacenados por
proceso. De esta manera no hace falta bajar las df grandes al PC, sino que podemos bajar ya las df que usamos para el analisis.

NOTE: los path necesarios ya estan correctamente escritos en este script con los pasth del cluster.
'''

import os
import pandas as pd
import json
import bz2
import base64
import os
import numpy as np
from tqdm import tqdm
import numpy as np


class DataGenerator:
    '''
    Las funciones de esta clase permiten añadir a bases de datos que acumulan informacion de diferentes procesos:
    - Evoluciones de: truth, estimaciones, degradaciones, precisiones o costes
    - Limites de regiones de aprendizaje
    - Evolucion de metricas para analizar la estabilidad de las estimaciones truth en funcion del numero de episodios
    '''

    def __init__(self,pack,seeds,
                list_freq,list_n_ep=[500,250,100,50,25,5], list_cost_perc=[0.05,0.1,0.15,0.2,0.25], # la lista de frecuanzias depende del numero de iteraciones con que se ejecute cada pack
                global_deg_metric='norm_from_mean_worsening_to_improvement',local_deg_metric='reward_diff',prec_metric='relative_perc_criteria_best', limit_metric='from_first_last',
                last_estimates_conf=100,
                data_common_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")):
        
        data_path=data_common_path+'/'+pack+'/analysis_data/'

        # Generar base de datos en donde se guardaran configuraciones por defecto y optimas de los criterios por pack, y los errores de las learning curves
        # df_conf = self.read_generate_df(data_common_path+'/configurations.csv',['pack','train_default','train_opt','test_default','test_opt','test_cost_opt'])
        # df_err=self.read_generate_df(data_common_path+'/errors.csv',['pack','last','train_default','test_default','test_cost_opt'])
        # df_conf.loc[len(df_conf)] = [pack, None, None, None, None, None]
        # df_err.loc[len(df_err)] = [pack, None, None, None, None]
        # df_conf.to_csv(data_common_path+'/configurations.csv', index=False)
        # df_err.to_csv(data_common_path+'/errors.csv', index=False)   

        # Primero generar las bases de datos de estimadores en la misma carpeta del cluster para cada proceso
        init,algo,env=pack.split('_')
        for i in tqdm(range(len(seeds))):
            # ProcessEstimator.compute_estimates(init+'_'+algo,env,seeds[i],'',None,'SE',data_path=data_path)
            # ProcessEstimator.compute_estimates(init+'_'+algo,env,seeds[i],'',None,'mean_diff',data_path=data_path)
            # ProcessEstimator.compute_estimates(init+'_'+algo,env,seeds[i],'',None,'CI_width',data_path=data_path)
            for n_ep in list_n_ep:
                ProcessEstimator.compute_estimates(init+'_'+algo,env,seeds[i],'',n_ep,'train')
                ProcessEstimator.compute_estimates(init+'_'+algo,env,seeds[i],'',n_ep,'test')
            

        # Despues generar las bases de datos necesarias para construir todas las graficas del analisis completo por pack
        # Necesitamos: limites de regiones de aprendizaje por proceso.
        # Necesitamos evoluciones de: degradacion, truth, truth de criterios por defecto, truth de criterio test con coste,
        # precision de criterios, coste de criterio test.

        for i in tqdm(range(len(seeds))):

            self.add_learning_limits_to_csv(data_path+'learning_regions.csv',pack,seeds[i],limit_metric=limit_metric)
            
            self.add_deg_to_csv(data_path+'deg_evolution.csv',pack,seeds[i],global_deg_metric=global_deg_metric,local_deg_metric=local_deg_metric)

            self.add_truth_to_csv(data_path+'df_best_truth.csv',pack,seeds[i],criteria='truth_best')
            self.add_truth_to_csv(data_path+'df_worst_truth.csv',pack,seeds[i],criteria='worst')

            self.add_truth_to_csv(data_path+'df_last_truth.csv',pack,seeds[i],criteria='last')
            # self.add_estimates_to_csv(data_path+'df_last_est.csv',pack,seeds[i],criteria='last',conf=[last_estimates_conf,None])
            self.add_criteria_prec_cost_to_csv(data_path+'df_last_prec.csv',pack,seeds[i],criteria='last')

            for n_ep in list_n_ep:
                self.add_truth_to_csv(data_path+'df_train_truth.csv',pack,seeds[i],conf=[n_ep,None],criteria='best_train')
                # self.add_estimates_to_csv(data_path+'df_train_est.csv',pack,seeds[i],conf=[n_ep,None],criteria='best_train')
                self.add_criteria_prec_cost_to_csv(data_path+'df_train_prec.csv',pack,seeds[i],conf=[n_ep,None,None],criteria='best_train')

            for freq in list_freq:
                for n_ep in list_n_ep:
                   self.add_truth_to_csv(data_path+'df_test_truth.csv',pack,seeds[i],conf=[n_ep,freq],criteria='best_val')
                #    self.add_estimates_to_csv(data_path+'df_test_est.csv',pack,seeds[i],conf=[n_ep,freq],criteria='best_val')
                   self.add_criteria_prec_cost_to_csv([data_path+'df_test_prec.csv',data_path+'df_test_cost.csv'],pack,seeds[i],conf=[n_ep,freq,None],criteria='best_val')
                for cost_perc in list_cost_perc:
                    self.add_criteria_prec_cost_to_csv([data_path+'df_test_prec.csv',data_path+'df_test_cost.csv',data_path+'df_test_n_ep.csv'],pack,seeds[i],conf= [None,freq,cost_perc],criteria='best_val_with_cost')
                    #self.add_truth_to_csv(data_path+'df_test_truth.csv',self.pack,self.seeds[i],conf= [None,freq],criteria='best_val_with_cost',cost_perc=cost_perc)

        # Guardar variable
        self.data_path=data_path
        self.data_source_path=data_common_path+'/'+pack+'/'
        self.seeds=seeds
        self.pack=pack

    def add_test_cost_truth(self,optimal_cost,optimal_freq):

        for i in tqdm(range(len(self.seeds))):
            self.add_truth_to_csv(self.data_path+'df_test_truth.csv',self.pack,self.seeds[i],conf= [None,optimal_freq],criteria='best_val_with_cost',cost_perc=optimal_cost)
        
    def add_truth_to_csv(self,path,pack,seed,conf=[None,None],criteria='truth_best',cost_perc=0.1):
        '''
        Añadir (si no existe) una columna con el truth de las politicas seleccionadas por los criterios.

        `path`: directorio asociado al .csv en donde queremos añadir una nueva evolucion de truth asociada a un proceso
        `criteria`: puede ser 'truth_best', 'worst', 'last', 'best_train', 'best_val', 'best_val_with_cost'
        '''

        # Extraer informacion de variables "comprimidas"
        n_ep,freq=conf

        # Determinar el label de la configuracion dependiendo del criterio que se usara para el nombre de la columna
        if criteria in ['last','truth_best','worst']:
            conf=''
        if n_ep!=None and freq==None:
            conf='_'+str(n_ep)
        if n_ep!=None and freq!=None:
            conf='_'+str(n_ep)+'_'+str(freq)
        if criteria=='best_val_with_cost':
            conf='_'+str(cost_perc)+'cost'+'_'+str(conf[1])

        # Inicializar clase que permite calcular las evoluciones de truth
        generator=EvolutionGenerator(pack,seed) # TODO: aqui si perc_time_start=0 da error, por eso hay que forzarlo despues de inicializar
        x_times=generator.estimator.df_train['time_seconds'].tolist()
        
        # Leer/generar csv para almacenar la nueva evolucion de truth si no esta ya almacenada
        df=ProcessEstimator.read_create_estimates_csv(path,generator.estimator.df_train.shape[0])
        if pack+str(seed)+str(conf) not in df.columns:
            truth_evol=generator.truth_evolution(x_times,n_ep=n_ep,freq=x_times[::freq],criteria=criteria,cost_perc=cost_perc)
            df[pack+str(seed)+str(conf)]=truth_evol
            df.to_csv(path,index=False)
 
    def add_estimates_to_csv(self,path,pack,seed,conf=[None,None],criteria='truth_best',cost_perc=0.1):

        '''
        Añadir (si no existe) una columna con las estimaciones de las politicas seleccionadas por los criterios.

        `criteria`: puede ser 'best_train', 'best_val', 'best_val_with_cost'

        NOTE: esta funcion la defino con la idea de representar las curvas de aprendizaje que se estan mostrando en la
        practica experimental. En la practica experimental no creo que las learning curves se muestren con el truth, sino
        con los valores estimados del truth para cada politica output en cada iteracion. Sin embargo, para nuestro analisis,
        lo que nos interesa es mostrar la influencia de los criterios conociendo la verdad absoluta. Por tanto, la evolucion de
        estos estimadores para las learning curves no nos sirve para nuestra motivacion.

        '''

        # Extraer informacion de variables "comprimidas"
        n_ep,freq=conf

        # Determinar el label de la configuracion dependiendo del criterio que se usara para el nombre de la columna
        if n_ep==None:
            conf=''
        if n_ep!=None and freq==None:
            conf='_'+str(n_ep)
        if n_ep!=None and freq!=None:
            conf='_'+str(n_ep)+'_'+str(freq)
        if criteria=='best_val_with_cost':
            conf='_'+str(cost_perc)+'cost'

        # Inicializar clase que permite calcular las evoluciones de estimaciones
        generator=EvolutionGenerator(pack,seed)
        x_times=generator.estimator.df_train['time_seconds'].tolist()
        
        # Leer/generar csv para almacenar la nueva evolucion de estimaciones si no esta ya almacenada
        df=ProcessEstimator.read_create_estimates_csv(path,generator.estimator.df_train.shape[0])
        if criteria!='last':
            if pack+str(seed)+str(conf) not in df.columns:
                estimation_evol=generator.estimation_evolution(x_times,n_ep=n_ep,freq=x_times[::freq],criteria=criteria,cost_perc=cost_perc)
                df[pack+str(seed)+str(conf)]=estimation_evol
                df.to_csv(path,index=False)
        else: # Last es un criterio que no considera estimaciones, por tanto para representar su curva con estimaciones, considero que usa las asociadas a la ultima politica con train default
            if pack+str(seed)+str(conf) not in df.columns:
                current_path=self.data_source_path+pack+'_seed'+str(seed)+'_/process_info/'
                df_train_estimates=pd.read_csv(current_path+'/df_traj_estimates.csv')
                last=np.array(df_train_estimates[conf.replace('_','')+'_traj_ep'].tolist())
                df[pack+str(seed)+str(conf)]=last
                df.to_csv(path,index=False)
                  
    def add_criteria_prec_cost_to_csv(self,path,pack,seed,conf=[None,None,None],criteria='truth_best',metric='relative_perc_criteria_best'):
        
        '''
        Añadir (si no existe) una columna con la precision y coste de las politicas seleccionadas por los criterios.

        `path`: directorio asociado al .csv en donde queremos añadir una nueva evolucion de precisiones y costes asociada a un proceso.
        En el caso de `criteria` 'best_val' o 'best_val_with_cost' tendra esta extructura: path=[path1,path2], con el csv de las precisiones y de los costes, respectivamente.
        `criteria`: puede ser 'truth_best', 'worst', 'last', 'best_train', 'best_val', 'best_val_with_cost'
        '''
        
        # Extraer informacion de variables "comprimidas"
        n_ep,freq,cost_perc=conf

        # Determinar el label de la configuracion dependiendo del criterio que se usara para el nombre de la columna
        if criteria=='last':
            conf=''
        if n_ep!=None and freq==None:
            conf='_'+str(n_ep)
        if n_ep!=None and freq!=None:
            conf='_'+str(n_ep)+'_'+str(freq)
        if criteria=='best_val_with_cost':
            conf='_'+str(cost_perc)+'cost_'+str(conf[1])

        # Inicializar clase que permite calcular las evoluciones 
        generator=EvolutionGenerator(pack,seed)
        x_times=generator.estimator.df_train['time_seconds'].tolist()
        
        # Leer/generar csv para almacenar la nueva evolucion si no esta ya almacenada
        if criteria in ['best_val']:
            path_prec,path_cost=path
            df_prec=ProcessEstimator.read_create_estimates_csv(path_prec,generator.estimator.df_train.shape[0])
            df_cost=ProcessEstimator.read_create_estimates_csv(path_cost,generator.estimator.df_train.shape[0])
            if pack+str(seed)+str(conf)+'_'+metric not in df_prec.columns:
                eff_evol,cost_evol=generator.effectiveness_evolution(x_times,n_ep=n_ep,freq=x_times[::freq],criteria=criteria,metric=metric,for_analyzer=True,cost_perc=cost_perc)
                df_prec[pack+str(seed)+str(conf)+'_'+metric]=eff_evol
                df_cost[pack+str(seed)+str(conf)+'_'+metric]=np.array(cost_evol) / np.array(x_times)
                df_prec.to_csv(path_prec,index=False)
                df_cost.to_csv(path_cost,index=False)

        if criteria in ['best_val_with_cost']:
            path_prec,path_cost,path_n_ep=path
            df_prec=ProcessEstimator.read_create_estimates_csv(path_prec,generator.estimator.df_train.shape[0])
            df_cost=ProcessEstimator.read_create_estimates_csv(path_cost,generator.estimator.df_train.shape[0])
            df_n_ep=ProcessEstimator.read_create_estimates_csv(path_n_ep,generator.estimator.df_train.shape[0])
            if pack+str(seed)+str(conf)+'_'+metric not in df_prec.columns:
                eff_evol,cost_evol,val_n_ep=generator.effectiveness_evolution(x_times,n_ep=n_ep,freq=x_times[::freq],criteria=criteria,metric=metric,for_analyzer=True,cost_perc=cost_perc)
                df_prec[pack+str(seed)+str(conf)+'_'+metric]=eff_evol
                df_cost[pack+str(seed)+str(conf)+'_'+metric]=np.array(cost_evol) / np.array(x_times)
                df_n_ep[pack+str(seed)+str(conf)+'_'+metric]=val_n_ep
                df_prec.to_csv(path_prec,index=False)
                df_cost.to_csv(path_cost,index=False)
                df_n_ep.to_csv(path_n_ep,index=False)

        if criteria in ['last','best_train']: 
            df=ProcessEstimator.read_create_estimates_csv(path,generator.estimator.df_train.shape[0])
            if pack+str(seed)+str(conf)+'_'+metric not in df.columns:
                eff_evol=generator.effectiveness_evolution(x_times,n_ep=n_ep,freq=None,criteria=criteria,metric=metric,for_analyzer=True,cost_perc=cost_perc)
                df[pack+str(seed)+str(conf)+'_'+metric]=eff_evol
                df.to_csv(path,index=False)

    def add_deg_to_csv(self,path,pack,seed,global_deg_metric='norm_worsening_to_improvement',local_deg_metric='reward_diff'):
        '''
        Obtener/añadir evolucion de degradacion asociada a un proceso.
        '''

        # Inicializar clase que permite calcular las evoluciones de degradacion
        generator=EvolutionGenerator(pack,seed)
        x_times=generator.estimator.df_train['time_seconds'].tolist()

        deg=generator.degradation_evolution(x_times,global_deg_metric,local_deg_metric)

        # Leer/generar csv para almacenar la nueva evolucion de degradacion si no esta ya almacenada
        df=ProcessEstimator.read_create_estimates_csv(path,generator.estimator.df_train.shape[0])
        if pack+str(seed)+'_'+global_deg_metric+'_'+local_deg_metric not in df.columns:
            df[pack+str(seed)+'_'+global_deg_metric+'_'+local_deg_metric]=deg
            df.to_csv(path,index=False)

    def add_learning_limits_to_csv(self,path,pack,seed,limit_metric):

        '''
        Obtener/añadir limites de regiones de aprendizaje.
        
        '''

        # Inicializar clase que permite calcular los limites 
        estimator=PointEstimator(pack,seed)
        a,b= estimator.learning_region_limits(limit_metric)

        # Leer/generar csv para almacenar los nuevos limites si no esta ya almacenada
        if not os.path.exists(path):
            df = pd.DataFrame(columns=['pack_seed','limit_metric','a','b','T'])
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_csv(path, index=False)
        else:
            df=pd.read_csv(path)

        if not ((df['pack_seed'] == pack+str(seed)) & (df['limit_metric'] == limit_metric)).any():
            df.loc[len(df)]=[pack+str(seed),limit_metric,a,b,estimator.df_test.shape[0]]
            df.to_csv(path,index=False)

    def read_generate_df(self,path_csv,column_names):

        if not os.path.exists(path_csv):
            df_estimates = pd.DataFrame(columns=column_names)
            os.makedirs(os.path.dirname(path_csv), exist_ok=True)
            df_estimates.to_csv(path_csv, index=False)
        else:
            df_estimates=pd.read_csv(path_csv)
        
        return df_estimates
    
class DataConverter:

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
        
    def from_df_data_to_graph_data(path,pack,
                                  which_graph=None,
                                  global_deg_metric=None,local_deg_metric=None,
                                  prec_metric=None,limit_metric=None,
                                  train_conf=None,test_conf=None,
                                  n_ep_type=None):
    
        '''
        Extrae la informacion pertinente de las bases de datos de degradacion, precision o coste por regiones de aprendizaje, 
        en el formato apropiado para generar a partir de esos datos las graficas de interes.
        '''
        
        df_limits=pd.read_csv(path[0])
        df_limits = df_limits[
                        df_limits['pack_seed'].str.contains(pack, na=False) &
                        df_limits['limit_metric'].str.contains(limit_metric, na=False)
                    ] # Solo filas del pack con la metrica de los limites indicada

        if which_graph in ['deg_distribution','last_prec']:

            df=pd.read_csv(path[1])
            df = df.filter(like=pack) # Solo columnas del pack

            if which_graph=='def_distribution':
                df = df.filter(regex=global_deg_metric+'_'+local_deg_metric+"$") # Solo columnas del pack con deg indicada
            if which_graph=='last_prec':
                df = df.filter(regex='_'+prec_metric+"$") # Solo columnas del pack con prec indicada

            # Almacenar degradaciones por region
            initialization,learning,stabilization=[],[],[]

            for pack_seed in tqdm(df_limits['pack_seed'],desc="Data for deg graphs"):
                a=int(df_limits.loc[df_limits['pack_seed'] == pack_seed, 'a'])
                b=int(df_limits.loc[df_limits['pack_seed'] == pack_seed, 'b'])

                df_pack_seed= df.filter(like=pack_seed+'_').iloc[:, 0].tolist()
                initialization+=df_pack_seed[:a]
                learning+=df_pack_seed[a:b]
                stabilization+=df_pack_seed[b:]

            return initialization,learning,stabilization
        
        if which_graph=='train_conf_prec':

            def get_conf_from_column_name(column_names):
                conf_list=[]
                for column_name in column_names:
                    splited=column_name.split('_')
                    conf_list.append(splited[3])

                return sorted(list(set(conf_list)), key=int)[::-1]

            df_prec=pd.read_csv(path[1])
            df_prec = df_prec.filter(like=pack) # Solo columnas del pack
            df_prec = df_prec.filter(regex=prec_metric+"$") # Solo columnas del pack con metrica de prec indicada

            # Almacenar datos por configuracion para cada region
            matrix_conf_initialization, matrix_conf_learning,matrix_conf_stabilization=[],[],[]

            conf_list=get_conf_from_column_name(df_prec.columns)

            for conf in tqdm(conf_list,desc="Data for train_prec graphs"):
                prec_initialization,prec_learning,prec_stabilization=[],[],[]

                df_prec_conf=df_prec.filter(like='_'+conf+'_')

                for pack_seed in df_limits['pack_seed']:
                    a=int(df_limits.loc[df_limits['pack_seed'] == pack_seed, 'a'])
                    b=int(df_limits.loc[df_limits['pack_seed'] == pack_seed, 'b'])

                    df_prec_conf_pack_seed= df_prec_conf.filter(like=pack_seed+'_').iloc[:, 0].tolist()
                    prec_initialization+=df_prec_conf_pack_seed[:a]
                    prec_learning+=df_prec_conf_pack_seed[a:b]
                    prec_stabilization+=df_prec_conf_pack_seed[b:]

                matrix_conf_initialization.append(prec_initialization)
                matrix_conf_learning.append(prec_learning)
                matrix_conf_stabilization.append(prec_stabilization)

            return matrix_conf_initialization,matrix_conf_learning,matrix_conf_stabilization,conf_list

        if which_graph=='test_conf_prec_cost':

            def get_conf_from_column_name(column_names,n_ep_type):
                
                n_ep_list,freq_list=[],[]
                for column_name in column_names:
                    splited=column_name.split('_')
                    if splited[4] not in ['','relative']: # TODO: ahora mismo en test_prec.csv hay algunas columnas mal guardadas, tendria que volver a generar ese csv y quitar la ultimas 2 condicion de este if
                        if n_ep_type=='with_cost':
                            n_ep_list.append(splited[3].replace('cost',''))
                        if n_ep_type=='constant':
                            n_ep_list.append(splited[3])
                        freq_list.append(splited[4])
    
                return sorted(list(set(n_ep_list)), key=float),sorted(list(set(freq_list)), key=int)

            df_prec=pd.read_csv(path[1])
            df_prec = df_prec.filter(like=pack) # Solo columnas del pack
            df_prec = df_prec.filter(regex=prec_metric+"$") # Solo columnas del pack con metrica de prec indicada

            if n_ep_type=='with_cost':
                df_prec=df_prec.loc[:, df_prec.columns.str.contains("cost")]
            if n_ep_type=='constant':
                df_prec=df_prec.loc[:, ~df_prec.columns.str.contains("cost")]

            # Almacenar datos por configuracion para cada region
            matrix1,matrix2,matrix3=[],[],[]
            n_ep_list,freq_list=get_conf_from_column_name(df_prec.columns,n_ep_type)
            for n_ep in tqdm(n_ep_list,desc="Data for test_prec_cost graphs"):
                n_ep_prec_initialization,n_ep_prec_learning,n_ep_prec_stabilization=[],[],[]

                for freq in freq_list:
                    freq_prec_initialization,freq_prec_learning,freq_prec_stabilization=[],[],[]

                    if n_ep_type=='constant':
                        df_prec_conf=df_prec.filter(like='_'+n_ep+'_'+freq+'_')
                    if n_ep_type=='with_cost':
                        df_prec_conf=df_prec.filter(like='_'+n_ep+'cost_'+freq+'_')

                    for pack_seed in df_limits['pack_seed']:
                        a=int(df_limits.loc[df_limits['pack_seed'] == pack_seed, 'a'])
                        b=int(df_limits.loc[df_limits['pack_seed'] == pack_seed, 'b'])

                        df_prec_conf_pack_seed= df_prec_conf.filter(like=pack_seed+'_').iloc[:, 0].tolist()
                        freq_prec_initialization+=df_prec_conf_pack_seed[:a]
                        freq_prec_learning+=df_prec_conf_pack_seed[a:b]
                        freq_prec_stabilization+=df_prec_conf_pack_seed[b:]

                    n_ep_prec_initialization.append(freq_prec_initialization)
                    n_ep_prec_learning.append(freq_prec_learning)
                    n_ep_prec_stabilization.append(freq_prec_stabilization)

                # Escribir aqui las listas para que no pete la memoria
                matrix1.append(n_ep_prec_initialization)
                matrix2.append(n_ep_prec_learning)
                matrix3.append(n_ep_prec_stabilization)

            return matrix1,matrix2,matrix3,n_ep_list,freq_list

        if which_graph=='how_times_best':
            # Quedarnos unicamente con los datos necesarios
            df_last=pd.read_csv(path[1])
            df_last = df_last.filter(regex=r''+pack) # Solo columnas asociadas al pack

            df_train=pd.read_csv(path[2])
            df_train = df_train.filter(regex=r''+pack+'.*_'+str(train_conf)+'$') # Solo columnas asociadas al pack y la configuracion indicada

            df_test=pd.read_csv(path[3])
            df_test = df_test.filter(regex=r''+pack+'.*_'+str(test_conf)+'$') # Solo columnas asociadas al pack y la configuracion indicada

            # Contar las veces que es cada par de criterios mejor por regiones (sin empates)
            last_train1,last_train2,last_train3=[0,0],[0,0],[0,0]
            last_test1,last_test2,last_test3=[0,0],[0,0],[0,0]
            train_test1,train_test2,train_test3=[0,0],[0,0],[0,0]
            len_init,len_learning,len_stabilization=0,0,0
            for pack_seed in df_limits['pack_seed']:

                def update_times_best_with_new_seed(truth1,truth2,a,b,old1,old2,old3):
                    truth1=np.array(truth1)
                    truth2=np.array(truth2)
                    olds=(np.array(old1),np.array(old2),np.array(old3))

                    news=[np.array([np.sum(truth1[:a]>truth2[:a]), np.sum(truth1[:a]<truth2[:a])]),
                            np.array([np.sum(truth1[a:b]>truth2[a:b]), np.sum(truth1[a:b]<truth2[a:b])]),
                            np.array([np.sum(truth1[b:]>truth2[b:]), np.sum(truth1[b:]<truth2[b:])])
                            ]

                    return [old+new for old,new in zip(olds, news)]
                
                a=int(df_limits.loc[df_limits['pack_seed'] == pack_seed, 'a'])
                b=int(df_limits.loc[df_limits['pack_seed'] == pack_seed, 'b'])

                truth_last=df_last.filter(like=pack_seed).iloc[:, 0].tolist()
                truth_train=df_train.filter(like=pack_seed+'_'+str(train_conf)).iloc[:, 0].tolist()
                last_train1,last_train2,last_train3 = update_times_best_with_new_seed(truth_last,truth_train,a,b,last_train1,last_train2,last_train3)

                truth_test=df_test.filter(like=pack_seed+'_'+str(test_conf)).iloc[:, 0].tolist()
                last_test1,last_test2,last_test3 = update_times_best_with_new_seed(truth_last,truth_test,a,b,last_test1,last_test2,last_test3)

                train_test1,train_test2,train_test3 = update_times_best_with_new_seed(truth_train,truth_test,a,b,train_test1,train_test2,train_test3)

                len_init+=a
                len_learning+=b-a
                len_stabilization+=len(truth_last)-b

            # Pasar de numero de veces a procentage
            matrix1=np.array([last_train1,last_test1,train_test1])/df_limits['a'].sum()
            matrix2=np.array([last_train2,last_test2,train_test2])/(df_limits['b'] - df_limits['a']).sum()
            matrix3=np.array([last_train3,last_test3,train_test3])/(df_limits['T'] - df_limits['b']).sum()

            return matrix1,matrix2,matrix3,len_init,len_learning,len_stabilization

        if which_graph=='in_which_deg_best':

            # Quedarnos unicamente con los datos necesarios
            df_last=pd.read_csv(path[1])
            df_last = df_last.filter(regex=r''+pack) # Solo columnas asociadas al pack

            df_train=pd.read_csv(path[2])
            df_train = df_train.filter(regex=r''+pack+'.*_'+str(train_conf)+'$') # Solo columnas asociadas al pack y la configuracion indicada

            df_test=pd.read_csv(path[3])
            df_test = df_test.filter(regex=r''+pack+'.*_'+str(test_conf)+'$') # Solo columnas asociadas al pack y la configuracion indicada

            df_deg=pd.read_csv(path[4])
            df_deg = df_deg.filter(like=pack) # Solo columnas del pack
            df_deg = df_deg.filter(regex=global_deg_metric+'_'+local_deg_metric+"$") # Solo columnas del pack con deg indicada

            # Acumular las degradaciones en que cada uno de los criterios de cada par es mejor 
            last_train1,last_train2,last_train3=[[],[]],[[],[]],[[],[]]
            last_test1,last_test2,last_test3=[[],[]],[[],[]],[[],[]]
            train_test1,train_test2,train_test3=[[],[]],[[],[]],[[],[]]
            for pack_seed in df_limits['pack_seed']:

                def update_deg_best_with_new_seed(deg,truth1,truth2,a,b,old1,old2,old3):
                    truth1=np.array(truth1)
                    truth2=np.array(truth2)
                    deg=np.array(deg)
                    olds=[old1,old2,old3]

                    news=[[deg[:a][truth1[:a] > truth2[:a]],deg[:a][truth1[:a] < truth2[:a]]],
                    [deg[a:b][truth1[a:b] > truth2[a:b]],deg[a:b][truth1[a:b] < truth2[a:b]]],
                    [deg[b:][truth1[b:] > truth2[b:]],deg[b:][truth1[b:] < truth2[b:]]]
                    ]

                    return [[list(sub_old) + list(sub_new) for sub_old, sub_new in zip(old_i, new_i)] for old_i, new_i in zip(olds, news)]
                
                a=int(df_limits.loc[df_limits['pack_seed'] == pack_seed, 'a'])
                b=int(df_limits.loc[df_limits['pack_seed'] == pack_seed, 'b'])
                deg=df_deg.filter(like=pack_seed).iloc[:, 0].tolist()


                truth_last=df_last.filter(like=pack_seed).iloc[:, 0].tolist()
                truth_train=df_train.filter(like=pack_seed+'_'+str(train_conf)).iloc[:, 0].tolist()
                last_train1,last_train2,last_train3 = update_deg_best_with_new_seed(deg,truth_last,truth_train,a,b,last_train1,last_train2,last_train3)

                truth_test=df_test.filter(like=pack_seed+'_'+str(test_conf)).iloc[:, 0].tolist()
                last_test1,last_test2,last_test3 = update_deg_best_with_new_seed(deg,truth_last,truth_test,a,b,last_test1,last_test2,last_test3)

                train_test1,train_test2,train_test3 = update_deg_best_with_new_seed(deg,truth_train,truth_test,a,b,train_test1,train_test2,train_test3)

            return [last_train1,last_test1,train_test1], [last_train2,last_test2,train_test2], [last_train3,last_test3,train_test3]

        if which_graph=='with_what_prec_diff_best':
            # Quedarnos unicamente con los datos necesarios
            df_last=pd.read_csv(path[1])
            df_last = df_last.filter(regex=r''+pack) # Solo columnas asociadas al pack

            df_train=pd.read_csv(path[2])
            df_train = df_train.filter(regex=r''+pack+'.*_'+str(train_conf)+'$') # Solo columnas asociadas al pack y la configuracion indicada

            df_test=pd.read_csv(path[3])
            df_test = df_test.filter(regex=r''+pack+'.*_'+str(test_conf)+'$') # Solo columnas asociadas al pack y la configuracion indicada

            df_prec_last=pd.read_csv(path[4])
            df_prec_last = df_prec_last.filter(regex=r''+pack+'.*_'+prec_metric+'$') # Solo columnas del pack con metrica de prec indicada

            df_prec_train=pd.read_csv(path[5])
            df_prec_train = df_prec_train.filter(regex=r''+pack+'.*_'+str(train_conf)+'_'+prec_metric+'$') # Solo columnas del pack con metrica de prec indicada

            df_prec_test=pd.read_csv(path[6])
            df_prec_test = df_prec_test.filter(regex=r''+pack+'.*_'+str(test_conf)+'_'+prec_metric+'$') # Solo columnas del pack con metrica de prec indicada

            # Acumular las precisiones en que cada uno de los criterios de cada par es mejor 
            last_train1,last_train2,last_train3=[[[],[]],[[],[]]],[[[],[]],[[],[]]],[[[],[]],[[],[]]]
            last_test1,last_test2,last_test3=[[[],[]],[[],[]]],[[[],[]],[[],[]]],[[[],[]],[[],[]]]
            train_test1,train_test2,train_test3=[[[],[]],[[],[]]],[[[],[]],[[],[]]],[[[],[]],[[],[]]]
            for pack_seed in df_limits['pack_seed']:

                def update_prec_best_with_new_seed(prec1,prec2,truth1,truth2,a,b,old1,old2,old3):
                    truth1=np.array(truth1)
                    truth2=np.array(truth2)
                    prec1=np.array(prec1)
                    prec2=np.array(prec2)
                    olds=[old1,old2,old3]

                    news=[[[prec1[:a][truth1[:a] > truth2[:a]],prec2[:a][truth1[:a] > truth2[:a]]],[prec2[:a][truth1[:a] < truth2[:a]],prec1[:a][truth1[:a] < truth2[:a]]]],
                    [[prec1[a:b][truth1[a:b] > truth2[a:b]],prec2[a:b][truth1[a:b] > truth2[a:b]]],[prec2[a:b][truth1[a:b] < truth2[a:b]],prec1[a:b][truth1[a:b] < truth2[a:b]]]],
                    [[prec1[b:][truth1[b:] > truth2[b:]],prec2[b:][truth1[b:] > truth2[b:]]],[prec2[b:][truth1[b:] < truth2[b:]],prec1[b:][truth1[b:] < truth2[b:]]]]
                    ]

                    return [[[list(x) + list(y) 
                        for x,y in zip(a_ij, b_ij)] 
                        for a_ij, b_ij in zip(a_i, b_i)] 
                        for a_i, b_i in zip(olds, news)]
                
                a=int(df_limits.loc[df_limits['pack_seed'] == pack_seed, 'a'])
                b=int(df_limits.loc[df_limits['pack_seed'] == pack_seed, 'b'])
                prec_last=df_prec_last.filter(like=pack_seed+'_'+prec_metric).iloc[:, 0].tolist()
                prec_train=df_prec_train.filter(like=pack_seed+'_'+str(train_conf)+'_'+prec_metric).iloc[:, 0].tolist()
                prec_test=df_prec_test.filter(like=pack_seed+'_'+str(test_conf)+'_'+prec_metric).iloc[:, 0].tolist()


                truth_last=df_last.filter(like=pack_seed).iloc[:, 0].tolist()
                truth_train=df_train.filter(like=pack_seed+'_'+str(train_conf)).iloc[:, 0].tolist()
                last_train1,last_train2,last_train3 = update_prec_best_with_new_seed(prec_last,prec_train,truth_last,truth_train,a,b,last_train1,last_train2,last_train3)

                truth_test=df_test.filter(like=pack_seed+'_'+str(test_conf)).iloc[:, 0].tolist()
                last_test1,last_test2,last_test3 = update_prec_best_with_new_seed(prec_last,prec_test,truth_last,truth_test,a,b,last_test1,last_test2,last_test3)

                train_test1,train_test2,train_test3 = update_prec_best_with_new_seed(prec_train,prec_test,truth_train,truth_test,a,b,train_test1,train_test2,train_test3)

            return [last_train1,last_test1,train_test1], [last_train2,last_test2,train_test2], [last_train3,last_test3,train_test3]

class ProcessEstimator:
    '''
    Las funciones de esta clase permiten hacer estimaciones asociadas a cada iteracion de un proceso a partir de los datos train o test
    almacenados durante la ejecucion de un proceso definido por un pack. Esas estimaciones pueden ser.
    '''

    # Relacionado con proceso completo
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

    def compute_estimates(algo,env,seed,resources,n_ep,train_test_estimate,data_path=None):
        '''
        Antes de generar las graficas, se calculan las estimaciones de expected episodic reward (EER) con train y test.
        Las estimaciones se almacenan en bases de datos adicionales. Asi se pueden calcular tantas estimaciones como
        curvas se quieran dibujar antes de dibujar las curvas (e.g. diferentes numeros de episodios para la estimacion train).
        Esto permitira no tener que hacer los mismos calculos multiples veces.

        Cuando las estimaciones se hacen con datos de validacion, no solo se calculan las estimaciones de EER, tambien el
        tiempo adicional necesario en segundos.
        '''
        
        # Leer bases de datos del proceso de interes
        current_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")+'/'+algo+'_'+env+'/'+algo+'_'+env+'_seed'+str(seed)+'_/process_info'
        df_train=pd.read_csv(current_path+'/df_traj.csv')
        df_test=pd.read_csv(current_path+'/df_val.csv')
        
        # Calcular estimaciones que usaremos como ground truth con los primeros 500 episodios de validacion
        df_test['ep_rewards']=[ DataConverter.compress_decompress_list(i,compress=False) for i in df_test['ep_rewards']]
        df_val_estimates=ProcessEstimator.read_create_estimates_csv(current_path+'/df_val_estimates.csv',df_train.shape[0])
        #print([ np.mean(i[:500]) for i in df_test['ep_rewards'] ]==df_val_estimates['truth'].tolist()) NOTE: tras pasar un tiempo y volver a repetir la operacion de descomprension y media, los resultados salen diferentes
        
        if 'truth' not in df_val_estimates.columns.tolist():
            df_val_estimates['truth']=[ np.mean(i[:500]) for i in df_test['ep_rewards'] ]
            df_val_estimates.to_csv(current_path+'/df_val_estimates.csv', index=False)
        if 'truth_norm' not in df_val_estimates.columns.tolist():
            df_val_estimates['truth_norm']=[ (i-min(df_val_estimates['truth']))/(max(df_val_estimates['truth'])-min(df_val_estimates['truth'])) for i in df_val_estimates['truth'] ]
            df_val_estimates.to_csv(current_path+'/df_val_estimates.csv', index=False)

        # Calcular evolucion de SE del truth de cada politica a medida que aumentamos en numero de episodios
        if train_test_estimate=='SE':
            df_SE=ProcessEstimator.read_create_estimates_csv(data_path+'df_SE.csv',1000)
            for policy_id,ep_rewards in enumerate(df_test['ep_rewards']):
                df_SE['pack_'+algo+'_'+env+str(seed)+'policy'+str(policy_id)]=[np.std(ep_rewards[:i], ddof=1)/np.sqrt(i) for i in range(1,1001)]
                df_SE.to_csv(data_path+'df_SE.csv',index=False)
        if train_test_estimate=='mean_diff':
            df_mean_diff=ProcessEstimator.read_create_estimates_csv(data_path+'df_mean_diff.csv',1000-1)
            for policy_id,ep_rewards in enumerate(df_test['ep_rewards']):
                df_mean_diff['pack_'+algo+'_'+env+str(seed)+'policy'+str(policy_id)]=[abs(np.mean(ep_rewards[:i])-np.mean(ep_rewards[:(i-1)])) for i in range(2,1001)]
                df_mean_diff.to_csv(data_path+'df_mean_diff.csv',index=False)
        if train_test_estimate=='CI_width':
            df_CI=ProcessEstimator.read_create_estimates_csv(data_path+'df_CI_width.csv',1000)
            for policy_id,ep_rewards in enumerate(df_test['ep_rewards']):
                df_CI['pack_'+algo+'_'+env+str(seed)+'policy'+str(policy_id)]=[abs(np.percentile(ep_rewards[:i],95)-np.percentile(ep_rewards[:i],5)) for i in range(1,1001)]
                df_CI.to_csv(data_path+'df_CI_width.csv',index=False)

        # Calcular estimaciones a partir de datos de train
        df_traj_estimates=ProcessEstimator.read_create_estimates_csv(current_path+'/df_traj_estimates.csv',df_train.shape[0])
        if train_test_estimate=='train':
            # Añadir columnas de interes en df_traj_estimates (siempre que ya no esten calculadas previamente)
            df_train['traj_rewards']=[ DataConverter.compress_decompress_list(i,compress=False) for i in df_train['traj_rewards']]
            df_train['traj_ep_end']=[ DataConverter.compress_decompress_list(i,compress=False) for i in df_train['traj_ep_end']]
            rollout=np.array(df_train['traj_rewards'][0]).shape[1]
            
            if str(n_ep)+'_traj_ep' not in df_traj_estimates.columns.tolist():
                df_traj_estimates[str(n_ep)+'_traj_ep']=ProcessEstimator.estimate_from_traj(
                    DataConverter.concat_traj_seq(df_train['traj_rewards']),
                    DataConverter.concat_traj_seq(df_train['traj_ep_end']),
                    list(range(rollout,rollout*(df_train.shape[0]+1),rollout)),n_ep)
                df_traj_estimates.to_csv(current_path+'/df_traj_estimates.csv', index=False)

        # Calcular estimaciones a partir de datos de validacion
        if train_test_estimate=='test':
            # Añadir columnas de interes en df_val_estimates (siempre que ya no esten calculadas previamente)
            if str(n_ep)+'_val_ep' not in df_val_estimates.columns.tolist():
                estimates=[np.mean(i[500:(500+n_ep)]) for i in df_test['ep_rewards']]
                val_times=[]
                for i in range(df_test.shape[0]):
                    df_test_elapsed_val_time=DataConverter.compress_decompress_list(df_test['elapsed_val_time'][i],compress=False)
                    df_test_n_val_ep=DataConverter.compress_decompress_list(df_test['n_val_ep'][i],compress=False)
                    val_time_until_truth=df_test_elapsed_val_time[df_test_n_val_ep.index(500)] 
                    val_time=df_test_elapsed_val_time[df_test_n_val_ep.index(500+n_ep)] 
                    val_times.append(val_time-val_time_until_truth)
                          
                df_val_estimates[str(n_ep)+'_val_ep']=[DataConverter.compress_decompress_list(i) for i in zip(estimates,val_times)]
                df_val_estimates.to_csv(current_path+'/df_val_estimates.csv', index=False)
       
class PointEstimator:

    def __init__(self,pack,seed):
          
        # Leer bases de datos del proceso de interes
        current_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")+'/'+pack+'/'+pack+'_seed'+str(seed)+'_/process_info'
        df_test=pd.read_csv(current_path+'/df_val.csv')
        df_test_estimates=pd.read_csv(current_path+'/df_val_estimates.csv')
        df_train=pd.read_csv(current_path+'/df_traj.csv')
        df_train_estimates=pd.read_csv(current_path+'/df_traj_estimates.csv')

        self.df_test=df_test
        self.df_test_estimates=df_test_estimates
        self.df_train=df_train
        self.df_train_estimates=df_train_estimates
        self.selector=Selector(pack,seed)


    # Relacionado con calculo puntual
    def learning_region_limits(self,limit_metric='from_first_last'):

        truth_last=self.df_test_estimates['truth'].tolist()

        a,b=None,None
        for i in range(len(truth_last)):

            if limit_metric=='from_mean':
                if abs(truth_last[i]-np.mean(truth_last[:i+1]))<=np.std(truth_last[:i+1]):
                    if a==None or i-a<int(len(truth_last)*0.1):
                        a=i
                if abs(np.mean(truth_last[-i-1:])-truth_last[-1-i])<=np.std(truth_last[-i-1:]) :
                    if b==None or b-len(truth_last)+i<int(len(truth_last)*0.1):
                        b=len(truth_last)-i

            if limit_metric=='from_first_last':
                if abs(truth_last[i]-truth_last[0])<=np.std(truth_last[:i+1]) :
                    if a==None or i-a<int(len(truth_last)*0.1):
                        a=i
                if abs(truth_last[-1]-truth_last[-1-i])<=np.std(truth_last[-i-1:]) :
                    if b==None or b-len(truth_last)+i<int(len(truth_last)*0.1):
                        b=len(truth_last)-i

        # Para que siempre haya un minimo de datos en los intervalos [0,a] y [b,T]
        if a in list(range(int(0.1*len(truth_last)))):
            a=int(0.1*len(truth_last))
        if b in list(range(len(truth_last)-int(0.1*len(truth_last)),len(truth_last)+1)):
            b=len(truth_last)-int(0.1*len(truth_last))

        return a,b

    def estimate_any_degradation(self,A,B,degradation_metric,also_dominance=False,additionals=None):
        '''
        `degradation_metric`: 'greater_prob', 'paired_diff_probpos' ,'relative_reward_diff', 'reward_diff'
        
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
        
        
        if degradation_metric=='paired_diff_probpos':

            # Variable de diferencias pareadas
            paired_diffs=B-A

            # Proporcion de positivos en la diferencia ponderada
            degradation=np.mean(paired_diffs>0)

            return degradation
        
        if degradation_metric=='relative_reward_diff':
            # NOTE: esta metrica es la unica entre todas que no usa la distribucion (muestra) de los rewards, unicamente usa la media.
            #Por ello, para que la degradacion local sea un valor en [0,1] y comparable con otras degradaciones locales de otro par
            #de politicas asociado a otras tareas/procesos, debemos normalizarla/estandarizarla. Para ello, necesitamos conocer el reward maximo
            #y el reward minimo alcanzable en esa tarea (cosa que desconocemos). Como nosotros solo usaremos esta medida para comparar
            #diferentes pares de politicas en el mismo proceso, aproximaremos esos valores con el maximo y minimo observados en el proceso.
            #Por tanto, la  metrica resultante es equivalente a la magnitud.

            indicator=[1 if np.mean(B)>np.mean(A) else 0][0]
            relative_diff=abs(np.mean(A)-np.mean(B))/(additionals[1]-additionals[0])
            degradation=relative_diff*indicator

            return degradation
        
        if degradation_metric=='reward_diff':

            return [0 if additionals[1]==additionals[0] else abs(np.mean(A)-np.mean(B))/(max([np.mean(A),np.mean(B)])-additionals[0])][0]
   
    def degradation_level(self,elapsed_time,global_metric,local_metric):
        '''
        TODO: todas las metricas que no usan la variable aleatoria no tienen porque tener asignada una metrica local de degradacion,
        creo que se puede simplificar el codigo asociado a esas metricas globales
        TODO: tambien hay un monton de metricas que no uso. Deberia borrar todas las que finalmente no he usado en ningun momento.

        `global_metric`: 'mean_update_deg', 'weighted_mean_best_later_deg', 'best_last_deg', 'relative_worsening_to_improvement', 'norm_worsening_to_improvement'
        `local_metric`: 'greater_prob', 'paired_diff_probpos_meanpos', 'paired_diff_median', 'paired_diff_probpos', 'relative_reward_diff'
        '''

        if global_metric=='mean_update_deg':
            last_idx=self.selector.last_policy(elapsed_time)
            degradation_level=np.mean(self.df_test_estimates['update_deg_'+local_metric][int(self.df_test.shape[0]*.1)-1:last_idx+1])

        if global_metric=='weighted_mean_best_later_deg':
            # Indice de la mejor truth politica
            truth_best_idx=self.selector.truth_best_policy(elapsed_time)
            last_idx=self.selector.last_policy(elapsed_time)

            # Guardar: 1) Degradaciones entre cada politica posterior a la mejor y la mejor; 2) Numero de iteraciones (normalizadas) trasncurridas entre cada politica posterior a la mejor y la mejor
            degradations_best_laters=[]
            iter_diff_best_laters=[]
            best_ep_rewards=np.array(DataConverter.compress_decompress_list(self.df_test.loc[truth_best_idx,'ep_rewards'],compress=False)[:500])
            for later_idx in range(truth_best_idx+1,last_idx+1):
                later_ep_rewards=np.array(DataConverter.compress_decompress_list(self.df_test.loc[later_idx,'ep_rewards'],compress=False)[:500])
                degradations_best_laters.append(self.estimate_any_degradation(later_ep_rewards,best_ep_rewards,local_metric))
                iter_diff_best_laters.append(later_idx-truth_best_idx)
            iter_diff_best_laters=np.array(iter_diff_best_laters)/sum(iter_diff_best_laters)

            # Nivel de degradacion: media de las degradaciones ponderada por los numeros de iteraciones trasncurridos
            degradation_level=0
            for weight, deg in zip(iter_diff_best_laters,degradations_best_laters):
                degradation_level+=weight*deg
        
        if global_metric=='best_last_deg':
            # Indices de la ultima y mejor politica
            truth_best_idx=self.selector.truth_best_policy(elapsed_time)
            last_idx=self.selector.last_policy(elapsed_time)

            # Degradacion local entre ellas
            best_ep_rewards=np.array(DataConverter.compress_decompress_list(self.df_test.loc[truth_best_idx,'ep_rewards'],compress=False)[:500])
            last_ep_rewards=np.array(DataConverter.compress_decompress_list(self.df_test.loc[last_idx,'ep_rewards'],compress=False)[:500])
            degradation_level=self.estimate_any_degradation(last_ep_rewards,best_ep_rewards,local_metric)

        if global_metric=='relative_best_last_deg':
            # Indices de la ultima y mejor politica
            last_idx=self.last_policy(elapsed_time)
            init_idx=last_idx-self.start_iter+1
            truth_best_idx=self.df_test_estimates.loc[init_idx:(last_idx+1),'truth'].idxmax()

            # Degradacion local entre ellas
            best_ep_rewards=np.array(DataConverter.compress_decompress_list(self.df_test.loc[truth_best_idx,'ep_rewards'],compress=False)[:500])
            last_ep_rewards=np.array(DataConverter.compress_decompress_list(self.df_test.loc[last_idx,'ep_rewards'],compress=False)[:500])
            degradation_level=self.estimate_any_degradation(last_ep_rewards,best_ep_rewards,local_metric)
            
        if global_metric=='relative_worsening_to_improvement':
            # Indices de la ultima y mejor politica en el tiempo transcurrido
            last_idx=self.selector.last_policy(elapsed_time)
            init_idx=last_idx-self.start_iter+1
            truth_best_idx=self.df_test_estimates.loc[init_idx:(last_idx+1),'truth'].idxmax()

            # Identificar la peor y mejor politicas del proceso (para la normmalizacion y que la metrica local este en [0,1])
            process_worst_ep_truth=self.df_test_estimates['truth'].min()
            process_best_ep_truth=self.df_test_estimates['truth'].max()

            # Degradacion local entre: mejor-ultima y mejor-primera (de ventana)
            best_ep_truth=self.df_test_estimates.loc[truth_best_idx,'truth']
            last_ep_truth=self.df_test_estimates.loc[last_idx,'truth']
            init_ep_truth=self.df_test_estimates.loc[init_idx,'truth']

            best_last_deg=self.estimate_any_degradation(last_ep_truth,best_ep_truth,local_metric,additionals=[process_worst_ep_truth,process_best_ep_truth])
            best_init_deg=self.estimate_any_degradation(init_ep_truth,best_ep_truth,local_metric,additionals=[process_worst_ep_truth,process_best_ep_truth])
            indicator=[0 if init_ep_truth!=last_ep_truth else 1][0]

            degradation_level=best_last_deg/max(best_init_deg,best_last_deg,indicator)

        if global_metric=='worsening_to_improvement':
            # Indices de la ultima y mejor politica en el tiempo transcurrido
            truth_best_idx=self.selector.truth_best_policy(elapsed_time)
            last_idx=self.selector.last_policy(elapsed_time)

            # Identificar la peor y mejor politicas del proceso (para la normalizacion y que la metrica local este en [0,1])
            process_worst_ep_truth=self.df_test_estimates['truth'].min()
            process_best_ep_truth=self.df_test_estimates['truth'].max()

            # Degradacion local entre: mejor-ultima y mejor-primera
            best_ep_truth=self.df_test_estimates.loc[truth_best_idx,'truth']
            last_ep_truth=self.df_test_estimates.loc[last_idx,'truth']
            init_ep_truth=self.df_test_estimates.loc[self.start_iter-1,'truth']

            best_last_deg=self.estimate_any_degradation(last_ep_truth,best_ep_truth,local_metric,additionals=[process_worst_ep_truth,process_best_ep_truth])
            best_init_deg=self.estimate_any_degradation(init_ep_truth,best_ep_truth,local_metric,additionals=[process_worst_ep_truth,process_best_ep_truth])
            indicator=[0 if init_ep_truth!=last_ep_truth else 1][0]

            degradation_level=best_last_deg/max(best_init_deg,best_last_deg,indicator)

        if global_metric=='norm_worsening_to_improvement':
            # Indices de la ultima, mejor y peor politica en el tiempo transcurrido
            truth_best_idx=self.selector.truth_best_policy(elapsed_time)
            worst_idx=self.selector.worst_policy(elapsed_time)
            last_idx=self.selector.last_policy(elapsed_time)

            # Degradacion local entre: mejor-ultima y mejor-peor actual
            best_ep_truth=self.df_test_estimates.loc[truth_best_idx,'truth']
            worst_ep_truth=self.df_test_estimates.loc[worst_idx,'truth']
            last_ep_truth=self.df_test_estimates.loc[last_idx,'truth']


            best_worst_deg=self.estimate_any_degradation(worst_ep_truth,best_ep_truth,local_metric,additionals=[np.mean(worst_ep_truth),np.mean(best_ep_truth)])
            best_last_deg=self.estimate_any_degradation(last_ep_truth,best_ep_truth,local_metric,additionals=[np.mean(worst_ep_truth),np.mean(best_ep_truth)])
            indicator=[0 if best_ep_truth!=worst_ep_truth else 1][0]

            degradation_level=best_last_deg/max(best_worst_deg,indicator)

        if global_metric=='norm_from_mean_worsening_to_improvement':
            # Indices de la ultima, mejor y peor politica en el tiempo transcurrido
            truth_best_idx=self.selector.truth_best_policy(elapsed_time)
            last_idx=self.selector.last_policy(elapsed_time)

            # Degradacion local entre: mejor-ultima y mejor-peor actual
            best_truth=self.df_test_estimates.loc[truth_best_idx,'truth']
            last_truth=self.df_test_estimates.loc[last_idx,'truth']
            current_mean_truth=min(self.df_test_estimates.loc[:last_idx,'truth'].mean(),last_truth)

            if best_truth!=last_truth:
                degradation_level=(best_truth-last_truth)/(best_truth-current_mean_truth)
            else:
                degradation_level=0

        return degradation_level
    
    def effectiveness(self,elapsed_time,n_policy,normalized,metric='relative_perc_criteria_best',val_time=0):

        '''
        Diferentes opciones para medir la precision de seleccion:  'relative_perc_criteria_best'

        Todas ellas estan definidas en [0,1], 1 es beuno y 0 malo.
        '''
            
        if metric=='relative_perc_criteria_best':

            last_policy=self.selector.last_policy(elapsed_time)
            last_policy_without_val=self.selector.last_policy(elapsed_time-val_time)
            if not normalized:
                EER_list=self.df_test_estimates[(self.df_test['n_policy']<=last_policy) ]['truth'].tolist()
                EER_list_without_val=self.df_test_estimates[(self.df_test['n_policy']<=last_policy_without_val) ]['truth'].tolist()
            if normalized:
                EER_list=self.df_test_estimates[(self.df_test['n_policy']<=last_policy) ]['truth_norm'].tolist()

            EER_real_best=max(EER_list)

            EER_criteria_best=EER_list[n_policy]

            if EER_criteria_best==min(EER_list_without_val):
                return 1
            else:
                return (EER_criteria_best-min(EER_list_without_val))/(EER_real_best-min(EER_list_without_val))
 
    
    def best_estimation_training(self,elapsed_time,n_traj_ep):

        # Lista de EER estimados con datos de train de la secunencia de politicas visitada hasta el momento
        estimated_EER_seq=self.df_train_estimates[(self.df_train['time_seconds']<=elapsed_time)][str(n_traj_ep)+'_traj_ep'].tolist()

        return max(estimated_EER_seq)

    def best_estimation_validation(self,elapsed_time,n_val_ep,freq):

        # Tiempos de validacion con frecuencia constante indicada
        current_val_times=[i for i in freq if i<=elapsed_time]

        # Indices de las politicas asociadas a esos tiempos, sus estimaciones de EER y el tiempo adicional consumido para su calculo
        current_val_policies=[]
        esti_time_seq=[]
        for time in current_val_times:
            policy_id=self.df_train.loc[(self.df_train['time_seconds']<=time) ].index.max()

            current_val_policies.append(policy_id) 
            esti_time_seq.append(self.df_test_estimates[self.df_test_estimates['n_policy']==policy_id][str(n_val_ep)+'_val_ep'].values[0])

        esti_time_seq= [DataConverter.compress_decompress_list(i,compress=False) for i in esti_time_seq]

        # Dividir estimaciones de tiempos adicionales
        estimated_EER_seq=[]
        times_seq=[]
        for estimation, time in esti_time_seq:
            estimated_EER_seq.append(estimation)
            times_seq.append(time)

        return max(estimated_EER_seq)
    
    def best_estimation_validation_with_cost(self,elapsed_time,cost_perc=0.1):
        # Lista de EER estimados con datos de test de la secunencia de politicas visitada hasta el momento
        estimated_EER_seq=self.df_test_estimates[(self.df_train['time_seconds']<=elapsed_time) ][str(cost_perc)+'cost_val_ep'].tolist()

        return max(estimated_EER_seq)

class Selector: 

    def __init__(self,pack,seed):
          
        # Leer bases de datos del proceso de interes
        current_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")+'/'+pack+'/'+pack+'_seed'+str(seed)+'_/process_info'
        df_train=pd.read_csv(current_path+'/df_traj.csv')
        df_train_estimates=pd.read_csv(current_path+'/df_traj_estimates.csv')
        df_test=pd.read_csv(current_path+'/df_val.csv')
        df_test_estimates=pd.read_csv(current_path+'/df_val_estimates.csv')

        self.df_train=df_train
        self.df_test=df_test
        self.df_train_estimates=df_train_estimates
        self.df_test_estimates=df_test_estimates

    def truth_best_policy(self,elapsed_time):
        policy_id=self.df_test_estimates[(self.df_train['time_seconds']<=elapsed_time)]['truth'].idxmax()
        return policy_id

    def worst_policy(self,elapsed_time):
        policy_id=self.df_test_estimates[(self.df_train['time_seconds']<=elapsed_time)]['truth'].idxmin()
        return policy_id

    def last_policy(self,elapsed_time):
        '''Dado el tiempo transcurrido de aprendizaje, devuelve el indice de la ultima politica visitada.'''
        last_policy=self.df_train[self.df_train['time_seconds']<=elapsed_time]['n_policy'].max()
        return last_policy

    def best_policy_training(self,elapsed_time,n_traj_ep):

        # Lista de EER estimados con datos de train de la secunencia de politicas visitada hasta el momento
        estimated_EER_seq=self.df_train_estimates[(self.df_train['time_seconds']<=elapsed_time)][str(n_traj_ep)+'_traj_ep'].tolist()

        # Indice de la politica con mayor mean ER en train
        return estimated_EER_seq.index(max(estimated_EER_seq)) 

    def best_policy_validation(self,elapsed_time,n_val_ep,freq):

        # Tiempos de validacion con frecuencia constante indicada
        current_val_times=[i for i in freq if i<=elapsed_time]

        # Indices de las politicas asociadas a esos tiempos, sus estimaciones de EER y el tiempo adicional consumido para su calculo
        current_val_policies=[]
        esti_time_seq=[]
        for time in current_val_times:
            policy_id=self.df_train.loc[(self.df_train['time_seconds']<=time) ].index.max()

            current_val_policies.append(policy_id) 
            esti_time_seq.append(self.df_test_estimates[self.df_test_estimates['n_policy']==policy_id][str(n_val_ep)+'_val_ep'].values[0])

        esti_time_seq= [DataConverter.compress_decompress_list(i,compress=False) for i in esti_time_seq]

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
    
    def best_policy_validation_with_cost(self,elapsed_time,cost_perc,freq):

        # Descomprimir columnas necesarias de df_test
        df_test_ep_rewards=[DataConverter.compress_decompress_list(i,compress=False) for i in self.df_test['ep_rewards']]
        df_test_elapsed_val_times=[DataConverter.compress_decompress_list(i,compress=False) for i in self.df_test['elapsed_val_time']]
        df_test_n_val_ep=[DataConverter.compress_decompress_list(i,compress=False) for i in self.df_test['n_val_ep']]

        # Seleccionar unicamente los datos asociados a los episodios reservados para test
        df_test_ep_rewards=[ep_rewards[500:] for ep_rewards in df_test_ep_rewards]

        # Tiempos de validacion con frecuencia constante indicada
        current_val_times=[i for i in freq if i<=elapsed_time]

        # Indices de las politicas asociadas a esos tiempos, sus estimaciones de EER y el tiempo adicional consumido para su calculo
        current_val_policies=[]
        esti_time_seq=[]
        val_n_ep=[0]*self.df_test.shape[0]# inicializamos con 0, porque solo tendran un n_ep>0 las politicas que toque validar
        validation_time=0 # lo necesito ir almacenando para calcular el n_ep de validacion que consume el coste que queremos
        for time in current_val_times:
            policy_id=self.df_train.loc[(self.df_train['time_seconds']<=time)].index.max()
            current_val_policies.append(policy_id) 

            val_time_until_truth=df_test_elapsed_val_times[policy_id][df_test_n_val_ep[policy_id].index(500)] 
            current_times=list(np.array(df_test_elapsed_val_times[policy_id])-val_time_until_truth)[500:]

            percentage=(np.array(current_times)+validation_time)/time
            index_best=next((i for i, val in reversed(list(enumerate(percentage))) if val < float(cost_perc)), 0)

            esti_time_seq.append([np.mean(df_test_ep_rewards[policy_id][:index_best+1]),current_times[index_best]])
            val_n_ep[policy_id]=index_best+1

            validation_time+=current_times[index_best]

        # Dividir estimaciones de tiempos adicionales
        estimated_EER_seq=[]
        times_seq=[]
        for estimation, time in esti_time_seq:
            estimated_EER_seq.append(estimation)
            times_seq.append(time)

        # Indice de la politica (en la subsecuencia de las politicas asociadas a las frecuencias) con mayor mean ER en validacion
        indx_subseq=estimated_EER_seq.index(max(estimated_EER_seq))


        # Politica seleccionada y tiempo extra total invertido en su seleccion
        return current_val_policies[indx_subseq], sum(times_seq),val_n_ep

class EvolutionGenerator:
    '''
    Esta clase contiene funciones que permiten generar la evolucion de diferentes metricas durante un proceso determinado por un pack-seed. 
    '''
    def __init__(self,pack,seed):

        self.selector=Selector(pack,seed)
        self.estimator=PointEstimator(pack,seed)


    def degradation_evolution(self,x_times,global_metric,local_metric):
        return [self.estimator.degradation_level(time,global_metric,local_metric) for time in x_times]

    def effectiveness_evolution(self,x_times,n_ep=None,freq=None,criteria='last',normalized=False,for_analyzer=False,
                                local_deg_metric=None,metric='perc_criteria_best',cost_perc=0.1):

        y_eff=[]
        x_extras=[]
        val_time=0
        for time in x_times:
            if criteria=='truth_best':
                policy_id=self.selector.truth_best_policy(time)
            if criteria=='worst':
                policy_id=self.selector.worst_policy(time)
            if criteria=='last':
                policy_id=self.selector.last_policy(time)
            if criteria=='best_train':
                policy_id=self.selector.best_policy_training(time,n_ep)
            if criteria=='best_val':
                policy_id,val_time=self.selector.best_policy_validation(time,n_ep,freq)
                x_extras.append(val_time)
            if criteria=='best_val_with_cost':
                policy_id,val_time,val_n_ep=self.selector.best_policy_validation_with_cost(time,cost_perc,freq)
                x_extras.append(val_time)

            if for_analyzer:
                val_time=0


            eff=self.estimator.effectiveness(time+val_time,policy_id,normalized=normalized,metric=metric,val_time=val_time)
            y_eff.append(eff)

        if criteria in ['best_val']:
            return y_eff, x_extras
        if criteria in ['best_val_with_cost']:
            return y_eff, x_extras, val_n_ep # para el ultimo elapsed_time es cuando se almacenan lod n_ep de todas las politicas validadas
        if criteria not in ['best_val','best_val_with_cost']:
            return y_eff
        
    def truth_evolution(self,x_times,n_ep=None,freq=None,criteria='last',cost_perc=0.1):

        '''
        Evolucion del expected episodic reward real de las politicas seleccionadas por el criterio indicado
        
        :param criteria: puede ser 'truth_best', 'worst', 'last', 'best_train', 'best_val', 'best_val_with_cost'
        '''

        y_truth=[]

        for time in x_times:
            if criteria=='truth_best':
                policy_id=self.selector.truth_best_policy(time)
            if criteria=='worst':
                policy_id=self.selector.worst_policy(time)
            if criteria=='last':
                policy_id=self.selector.last_policy(time)
            if criteria=='best_train':
                policy_id=self.selector.best_policy_training(time,n_ep)
            if criteria=='best_val':
                policy_id,_=self.selector.best_policy_validation(time,n_ep,freq)
            if criteria=='best_val_with_cost':
                policy_id,_,_=self.selector.best_policy_validation_with_cost(time,cost_perc,freq)
            
            y_truth.append(self.selector.df_test_estimates[self.selector.df_test_estimates['n_policy']==policy_id]['truth'].values[0])
        
        return y_truth

    def estimation_evolution(self,x_times,n_ep=None,freq=None,criteria='last',cost_perc=0.1):

        '''
        Evolucion del expected episodic reward estimado de las politicas seleccionadas por el criterio indicado
        
        :param criteria: puede ser 'truth_best', 'worst', 'last', 'best_train', 'best_val'
        '''

        y_estimation=[]

        for time in x_times:
            if criteria=='best_train':
                estimation=self.estimator.best_estimation_training(time,n_ep)
            if criteria=='best_val':
                estimation=self.estimator.best_estimation_validation(time,n_ep,freq)
            if criteria=='best_val_with_cost':
                estimation=self.estimator.best_estimation_validation_with_cost(time,cost_perc)
            
            y_estimation.append(estimation)
        
        return y_estimation  

if __name__ == "__main__":

    DataGenerator('pack_PPO_BipedalWalker',list(range(1,31)),[50,25,10,5,2,1])
    DataGenerator('pack_PPO_LunarLanderContinuous',list(range(1,31)),[40,20,10,5,2,1])
