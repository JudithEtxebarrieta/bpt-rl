from Main import *

class ComputeCommonData():
    def __init__(self,library,
                 all_seeds=[list(range(1,31))]*4,
                 all_packs=[
                            'pack_PPO_BipedalWalker',
                            'pack_PPO_LunarLanderContinuous',
                            'pack_PPO_Ant',
                            'pack_PPO_Walker2d'
                            # 'pack_PPO_HalfCheetah',
                            # 'pack_PPO_Hopper'
                            ]
                 ):
        
        for pack,seeds in zip(all_packs,all_seeds):
            datagenerator=DataGenerator(library,pack,seeds,generated_in_cluster=True)
            datagenerator.add_p5prec_by_conf_and_regions_per_pack(pack)

        DataConverter.from_p5prec_to_common_optimal_conf()

class ExecutePackAnalysis():

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
        ''' Generar las graficas para test con n_ep constante y test con n_ep regulado por el coste de validacion'''
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
        self.grapher.graph_pack_learning_curves_error(self.pack)
        

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
        self.grapher.graph_test_with_cost()


# Programa principal
# ComputeCommonData('SB3')


# analyzer=ExecutePackAnalysis('SB3','pack_PPO_BipedalWalker',list(range(1,31)),100,5,5)
# analyzer=ExecutePackAnalysis('SB3','pack_PPO_LunarLanderContinuous',list(range(1,31)),100,5,10)
# analyzer=ExecutePackAnalysis('SB3','pack_PPO_Walker2d',list(range(1,31)),100,5,20)
analyzer=ExecutePackAnalysis('SB3','pack_PPO_Ant',list(range(1,31)),100,5,5)

# analyzer=ExecutePackAnalysis('SB3','pack_PPO_HalfCheetah',[1,2,3,4,5,6,13,15,16,17,18,19,20,21,24,29],100,5,20)

if __name__ == "__main__":

    analyzer.main_analysis1()
    analyzer.main_analysis2()
    analyzer.main_motivation_recommendation()
    analyzer.discussion_analysis()
    analyzer.internal_analysis()


path='experiments/results/data/SB3_PPO_Walker2d/'
seeds=list(range(1,31))


def join_df_analysis_seed(path,seeds,only_test_with_cost_truth=False):
    
    if only_test_with_cost_truth:
        df = pd.concat([pd.read_csv(path+'df_test_truth_with_cost'+str(i)+'.csv') for i in seeds],axis=1)
        df = pd.concat([pd.read_csv(path+"df_test_truth.csv"),df ],axis=1)
        df.to_csv(path+"df_test_truth.csv", index=False)

        # Borrar archivos usados
        for f in [path+'df_test_truth_with_cost'+str(i)+'.csv' for i in seeds]:
            if os.path.exists(f):
                os.remove(f)

    else:
        list_csv_names=['deg_evolution','learning_regions',
            'df_best_truth','df_last_truth','df_train_truth','df_test_truth',
            'df_last_prec','df_train_prec','df_test_prec',
            'df_test_cost','df_test_n_ep'
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

# join_df_analysis_seed(path,seeds,only_test_with_cost_truth=True)



# Borrar archivos usados
# for f in [path+'df_test_truth_with_cost'+str(i)+'.csv' for i in seeds]:
#     if os.path.exists(f):
#         os.remove(f)

# else:
#     list_csv_names=['deg_evolution','learning_regions',
#         'df_best_truth','df_last_truth','df_train_truth','df_test_truth',
#         'df_last_prec','df_train_prec','df_test_prec',
#         'df_test_cost','df_test_n_ep'
#         ]

#     for csv_name in list_csv_names:
#         # Borrar archivos usados
#         for f in [path+csv_name+str(i)+".csv" for i in seeds]:
#             if os.path.exists(f):
#                 os.remove(f)


