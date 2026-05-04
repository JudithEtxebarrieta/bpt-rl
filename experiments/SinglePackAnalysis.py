from Main import *

class ComputeCommonData():
    '''Guardar la mejor configuracion de los criterios teniendo en cuenta todos los setups'''

    def __init__(self,library,
                 all_seeds=[list(range(1,31))]*9,
                 all_packs=['pack_PPO_Pendulum',
                            'pack_PPO_BipedalWalker',
                            'pack_PPO_BipedalWalkerNorm',
                            'pack_PPO_LunarLanderContinuous',
                            'pack_PPO_Swimmer',
                            'pack_PPO_Ant',
                            'pack_PPO_Walker2d',
                            'pack_PPO_HalfCheetah',
                            'pack_PPO_Hopper'
                            ]
                 ):

        # Almacenar p5 de precision de cada region para cada posible configuracion
        for pack,seeds in zip(all_packs,all_seeds):
            datagenerator=DataGenerator(library,pack,seeds,generated_in_cluster=True)
            datagenerator.add_p25prec_by_conf_and_regions_per_pack(pack)

        # Obtener y guardar la configuracion con mayor p5 de precision de promedio en todas las regiones
        DataConverter.from_p25prec_to_common_optimal_conf()

class PackAnalyzer():
    '''
    Generar graficas para el analisis principal, analisis internos, y analsis de discusion.

    Analisis principal completo para el setup indicado. Este analisis contiene:
    - Distribucion de degradacion por regiones
    - Precision de los tres criterios por regiones
    - Comparacion two-by-two de los tres criterios: cuantas veces mejor, en que degradaciones mejor, con que precision y deiferencia mejor
    - Curvas de aprendizaje con los tres criterios por defecto vs nuestra recomendacion

    Graficas internas, diseñadas para mostrar explicitamente:
    - Degradacion, curvas de aprendizaje y regiones por proceso determinado por cada seed
    (esto lo usamos para comprobar que las definiciones de degradacion y regiones sean correctas)
    - CI de las precisiones por region asociadas a cada posible configuracion considerada para cada criterio
    (esto lo usamos para seleccionar una configuracion por cada criterio comun en todos los setups)
    - Estabilizacion de estimaciones truth
    (por ahora es provisional, comentamos que las graficas actuales no son las mas apropiadas para mostrar que nuestra estimacion truth es estable)

    Graficas de discusion, diseñadas para mostrar explicitamente:
    - Coste de validacion limitado por el umbral indicado, y n_ep o freq definidas automaticamente por ese coste

    '''

    def __init__(self,library,pack,seeds,
                 train_default_n_ep,test_default_n_ep,test_default_freq,
                 global_deg_metric='norm_from_mean_worsening_to_improvement',local_deg_metric='reward_diff',
                 prec_metric='relative_perc_criteria_best',
                 ):

        # Generar datos necesarios para el analisis a partir de los datos almacenados para cada proceso
        # (esto lo hacemos en el cluster, aqui solo inicializamos la clase)
        self.datagenerator=DataGenerator(library,pack,seeds,generated_in_cluster=True)
        self.grapher=Grapher(library,pack.replace('pack_',''))

        # Base de datos para guardar las configuraciones
        self.df_conf = self.datagenerator.read_generate_df(self.datagenerator.data_common_path+'/configurations.csv',['pack','train_default','train_opt','test_default','test_opt','test_cost_opt'])

        # Las demas variables
        self.pack=pack
        self.seeds=seeds
        self.train_default_n_ep=train_default_n_ep
        self.test_default_n_ep=test_default_n_ep
        self.test_default_freq=test_default_freq
        self.global_deg_metric=global_deg_metric
        self.local_deg_metric=local_deg_metric
        self.prec_metric=prec_metric

    #Graficas principales
    def main_analysis1(self):
        ''' Generar las graficas para test con n_ep constante y test con n_ep o freq regulado por el coste de validacion'''
        self.grapher.graph_deg_criteria_conf_by_regions(self.pack,self.train_default_n_ep,self.test_default_n_ep,self.test_default_freq,global_deg_metric=self.global_deg_metric,local_deg_metric=self.local_deg_metric)
        self.grapher.graph_deg_criteria_conf_by_regions(self.pack,self.train_default_n_ep,self.test_default_n_ep,self.test_default_freq,
                                                        global_deg_metric=self.global_deg_metric,local_deg_metric=self.local_deg_metric,n_ep_type='with_cost_n_ep')
        self.grapher.graph_deg_criteria_conf_by_regions(self.pack,self.train_default_n_ep,self.test_default_n_ep,self.test_default_freq,
                                                        global_deg_metric=self.global_deg_metric,local_deg_metric=self.local_deg_metric,n_ep_type='with_cost_freq')

    def main_analysis2(self):
        ''' Generar analisis comparativo de mejores versiones de los criterios'''
        self.grapher.graph_criteria_comparison_by_regions(self.pack,global_deg_metric=self.global_deg_metric,local_deg_metric=self.local_deg_metric,
                                                    prec_metric=self.prec_metric)

    def main_motivation_recommendation(self):
        '''Recomendacion'''
        self.grapher.graph_pack_learning_curves_with_criteria(self.pack)


    # Graficas secundarias
    def internal_analysis(self):
        ''' Para mostrar como la degradacion y los limites definen lo que decimos'''
        self.grapher.graph_pack_all_truth_with_regions(self.pack,self.seeds,
                                                global_deg_metric='norm_from_mean_worsening_to_improvement')

        #----- Para ciertor packs tambien estan ejecutados
        # self.grapher.graph_pack_all_truth_with_regions(self.pack,self.seeds,
        #                                         global_deg_metric='best_last_deg',local_deg_metric='paired_diff_probpos')
        # self.grapher.graph_pack_all_truth_with_regions(self.pack,self.seeds,
        #                                         global_deg_metric='best_last_deg',local_deg_metric='greater_prob')

        # Para mostrar la estabilidad de la estimacion considerada para truth
        # self.grapher.graph_pack_all_stability_truth_estimator(self.pack,self.seeds)
        # self.grapher.graph_pack_all_stability_truth_estimator(self.pack,self.seeds,stability_metric='mean_diff')
        # self.grapher.graph_pack_all_stability_truth_estimator(self.pack,self.seeds,stability_metric='CI_width')


    # Graficas para discusion
    def discussion_analysis(self):
        # self.grapher.graph_test_with_cost()
        self.grapher.graph_early_stopping(self.pack)

class ExecutePackAnalysis():
    def __init__(self,library,pack,default_train_n_ep,default_test_n_ep,default_test_freq,seeds=list(range(1,31))):

        analyzer=PackAnalyzer(library,pack,seeds,default_train_n_ep,default_test_n_ep,default_test_freq)

        analyzer.main_analysis1()
        analyzer.main_analysis2()
        analyzer.main_motivation_recommendation()
        analyzer.discussion_analysis()
        analyzer.internal_analysis()

#==================================================================================================
# Programa principal
#==================================================================================================

# Obtener configuraciones a considerar para cada criterio
# ComputeCommonData('SB3')

# # Analisis para entornos ClassicControl
# ExecutePackAnalysis('SB3','pack_PPO_Pendulum',100,5,10)

# # Analisis para entornos Box2D
# ExecutePackAnalysis('SB3','pack_PPO_BipedalWalker',100,5,5)
# ExecutePackAnalysis('SB3','pack_PPO_LunarLanderContinuous',100,5,10)

# # Analisis para entornos MuJoCo
ExecutePackAnalysis('SB3','pack_PPO_Swimmer',100,5,10)
# ExecutePackAnalysis('SB3','pack_PPO_HalfCheetah',100,5,20)
# ExecutePackAnalysis('SB3','pack_PPO_Ant',100,5,5)
# ExecutePackAnalysis('SB3','pack_PPO_Hopper',100,5,20)
# ExecutePackAnalysis('SB3','pack_PPO_Walker2d',100,5,20)

# # Analisis de entornos normalizados
# ExecutePackAnalysis('SB3','pack_PPO_BipedalWalkerNorm',100,5,5)

path='experiments/results/data/SB3_PPO_Pendulum/'
# path='experiments/results/data/SB3_PPO_BipedalWalker/'
# path='experiments/results/data/SB3_PPO_BipedalWalkerNorm/'
# path='experiments/results/data/SB3_PPO_LunarLanderContinuous/'
# path='experiments/results/data/SB3_PPO_Swimmer/'
# path='experiments/results/data/SB3_PPO_Ant/'
# path='experiments/results/data/SB3_PPO_HalfCheetah/'
# path='experiments/results/data/SB3_PPO_Hopper/'
# path='experiments/results/data/SB3_PPO_Walker2d/'
seeds=list(range(1,31))


def join_df_analysis_seed(path,seeds,only_test_with_cost_truth=False,only_default_est=False):

    if only_test_with_cost_truth:
        df = pd.concat([pd.read_csv(path+'df_test_truth_with_cost'+str(i)+'.csv') for i in seeds],axis=1)
        df = pd.concat([pd.read_csv(path+"df_test_truth.csv"),df ],axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
        df.to_csv(path+"df_test_truth.csv", index=False)

        # Borrar archivos usados
        for f in [path+'df_test_truth_with_cost'+str(i)+'.csv' for i in seeds]:
            if os.path.exists(f):
                os.remove(f)

    elif only_default_est:
        df = pd.concat([pd.read_csv(path+'df_train_default_est'+str(i)+'.csv') for i in seeds],axis=1)
        df = pd.concat([pd.read_csv(path+"df_train_est.csv"),df ],axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
        df.to_csv(path+"df_train_est.csv", index=False)
        

        df = pd.concat([pd.read_csv(path+'df_test_default_est'+str(i)+'.csv') for i in seeds],axis=1)
        df = pd.concat([pd.read_csv(path+"df_test_est.csv"),df ],axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
        df.to_csv(path+"df_test_est.csv", index=False)

        # Borrar archivos usados
        for f in [path+'df_test_default_est'+str(i)+'.csv' for i in seeds]:
            if os.path.exists(f):
                os.remove(f)
        for f in [path+'df_train_default_est'+str(i)+'.csv' for i in seeds]:
            if os.path.exists(f):
                os.remove(f)

    else:
        list_csv_names=['deg_evolution','learning_regions',
            'df_best_truth','df_last_truth','df_train_truth','df_test_truth',
            'df_last_prec','df_train_prec','df_test_prec',
            'df_test_cost','df_test_n_ep',
            'df_last_eff','df_train_eff','df_test_eff',
            'df_last_est','df_train_est','df_test_est'
            ]


        for csv_name in list_csv_names:
            if csv_name!='learning_regions':
                dfs = [pd.read_csv(path+csv_name+str(i)+".csv") for i in seeds]
                for i in range(1,len(dfs)):
                    dfs[i] = dfs[i].drop(columns=['n_policy'], errors='ignore') # Para que la columna 'n_policy' no salga repetida

                df = pd.concat(dfs, axis=1)
                df.to_csv(path+csv_name+".csv", index=False)

            else:
                df = pd.concat([pd.read_csv(path+csv_name+str(i)+".csv") for i in seeds],ignore_index=True)
                df.to_csv(path+csv_name+".csv", index=False)

            # Borrar archivos usados
            for f in [path+csv_name+str(i)+".csv" for i in seeds]:
                if os.path.exists(f):
                    os.remove(f)

# join_df_analysis_seed(path,seeds,only_test_with_cost_truth=False)
# join_df_analysis_seed(path,seeds,only_test_with_cost_truth=True)
# join_df_analysis_seed(path,seeds,only_default_est=True)








