from Main import *

class ExecutePackAnalysis():

    def __init__(self,library,pack,seeds,
                 train_default_n_ep,test_default_n_ep,test_default_freq,
                 global_deg_metric='norm_from_mean_worsening_to_improvement',local_deg_metric='reward_diff',
                 prec_metric='relative_perc_criteria_best',
                 limit_metric='from_first_last'
                 ):
        
        # Generar datos necesarios para el analisis a partir de los datos almacenados para cada proceso 
        # (esto lo hacemos en el cluster, aqui solo inicializamos la clase)
        self.datagenerator=DataGenerator(library,pack,seeds)
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
        self.limit_metric=limit_metric
        
    #Graficas principales
    def main_analysis1(self):
        # Generar las graficas para test con n_ep constante y test con n_ep regulado por el coste de validacion
        n_ep_train,n_ep_test1,freq_test1=self.grapher.graph_deg_criteria_conf_by_regions(self.pack,self.train_default_n_ep,self.test_default_n_ep,self.test_default_freq,global_deg_metric=self.global_deg_metric,local_deg_metric=self.local_deg_metric)
        n_ep_train,n_ep_test2,freq_test2=self.grapher.graph_deg_criteria_conf_by_regions(self.pack,self.train_default_n_ep,self.test_default_n_ep,self.test_default_freq,global_deg_metric=self.global_deg_metric,local_deg_metric=self.local_deg_metric,n_ep_type='with_cost')
        
        # Guardar configuraciones optimas
        self.df_conf.loc[len(self.df_conf)] = [self.pack, self.train_default_n_ep, int(n_ep_train), str(self.test_default_n_ep)+'_'+str(self.test_default_freq), str(n_ep_test1)+'_'+str(freq_test1), str(n_ep_test2)+'_'+str(freq_test2)]
        self.df_conf.to_csv(self.datagenerator.data_common_path+'/configurations.csv', index=False)

        # Los datos truth del criterio test con coste optimo debemos almacenarlos antes de ejecutar el main_analysis2 
        # Ahora esto lo hago en el cluster tras ejecutar aqui main_analysis1 y conseguir la configuracion optima)
        #self.datagenerator.add_test_cost_truth(n_ep_test2,freq_test2)

    def main_analysis2(self):

        # Leer configuraciones optimas almacenadas
        n_ep_train,test_cost_opt = self.df_conf.loc[self.df_conf['pack'] == self.pack,['train_opt', 'test_cost_opt']].iloc[0]
        n_ep_test,freq_test=test_cost_opt.split('_')

        # Generar analisis comparativo de mejores versiones de los criterios
        self.grapher.graph_criteria_comparison_by_regions(self.pack,global_deg_metric=self.global_deg_metric,local_deg_metric=self.local_deg_metric,
                                                    prec_metric=self.prec_metric,limit_metric=self.limit_metric,
                                                    train_conf=int(n_ep_train),test_conf=str(n_ep_test)+'cost_'+str(freq_test))
        
    def main_motivation_recommendation(self):

        # Leer configuraciones optimas almacenadas
        n_ep_train,test_cost_opt = self.df_conf.loc[self.df_conf['pack'] == self.pack,['train_opt', 'test_cost_opt']].iloc[0]
        n_ep_test,freq_test=test_cost_opt.split('_')

        n_ep_test,freq_test=None,None
        # Recomendacion
        self.grapher.graph_pack_learning_curves_with_criteria(self.pack,default_conf=[self.train_default_n_ep,self.test_default_n_ep,self.test_default_freq],optimal_conf=str(n_ep_test)+'cost_'+str(freq_test),also_train_test_grid=False) #default_test_n_ep=eval_freq/n_steps
        
        #----- Para ciertor packs tambien estan ejecutados
        # self.grapher.graph_pack_learning_curves_with_criteria(self.pack,default_conf=[self.train_default_n_ep,self.test_default_n_ep,self.test_default_freq],optimal_conf=str(n_ep_test)+'cost_'+str(freq_test),curves='estimate_truth')
        # self.grapher.graph_pack_learning_curves_error(self.pack,default_conf=[self.train_default_n_ep,self.test_default_n_ep,self.test_default_freq],diff='truth',optimal_conf=str(n_ep_test)+'cost_'+str(freq_test))
        # self.grapher.graph_pack_learning_curves_error(self.pack,default_conf=[self.train_default_n_ep,self.test_default_n_ep,self.test_default_freq],optimal_conf=str(n_ep_test)+'cost_'+str(freq_test))

    # Graficas secundarias
    def internal_analysis(self):
        # Para mostrar como la degradacion y los limites definen lo que decimos
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
        self.grapher.graph_test_with_cost_n_ep()


# Programa principal

# analyzer=ExecutePackAnalysis('SB3','pack_PPO_BipedalWalker',list(range(1,31)),100,5,5)
# analyzer=ExecutePackAnalysis('SB3','pack_PPO_LunarLanderContinuous',list(range(1,31)),100,5,10)
# analyzer=ExecutePackAnalysis('SB3','pack_PPO_Walker2d',[2,9,10,11,12,13,14,15,17,18,19,20,22,23,24],100,5,20)
analyzer=ExecutePackAnalysis('SB3','pack_PPO_Ant',[1,2,3,4,7,8,9,17,18],100,5,5)

if __name__ == "__main__":

    analyzer.main_analysis1()
    # analyzer.main_analysis2()
    analyzer.main_motivation_recommendation()
    analyzer.discussion_analysis()
    analyzer.internal_analysis()

'''
path='experiments/results/data/SB3_PPO_Walker2d/'
seeds=[2,9,10,11,12,13,14,15,17,18,19,20,22,23,24]#list(range(1,31))

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
        # Borrar archivos usados
        for f in [path+csv_name+str(i)+".csv" for i in seeds]:
            if os.path.exists(f):
                os.remove(f)
'''





