from tqdm import tqdm
from Main import Converter, Estimator, ProcessIndependentAnalyzer   

# EXPERIMENTOS
experiments_intuition=False
new_experiments=False
experiments_when_which=False

# GENERAL
train_n_ep=[500]# n_ep mas comun entre todos los procesos de todos los entornos despues del tuning (criteria_conf_by_process.csv)
test_n_ep=[32]# n_ep minimo del maximo que no supera el 25% de tiempo en validar entre todos los procesos de todos los entornos (test_affordable_n_ep_by_process.csv)
test_freq=[1]# freq mas comun entre todos los procesos de todos los entornos despues del tuning (criteria_conf_by_process.csv)

halfcheetah_process_ids=['PPO_HalfCheetah_seed'+str(seed) for seed in range(1,11)]
walker2d_process_ids=['PPO_Walker2d_seed'+str(seed) for seed in range(1,11)]
ant_process_ids=['PPO_Ant_seed'+str(seed) for seed in range(1,11)]
humanoid_process_ids=['PPO_Humanoid_seed'+str(seed) for seed in range(1,11)]
humanidstandup_process_ids=['PPO_HumanoidStandup_seed'+str(seed) for seed in range(1,11)]

#==================================================================================================
# Experimentos realizados para ganar intuicion 
# (esta experimentación esta resumida cronologicamente en experiments_intuition/README_intuition/main.pdf )
#==================================================================================================

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

if experiments_intuition:

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
    grid_train_n_ep=[500,250,100,50,25] # Numeros de episodios optimos observados para train
    grid_test_n_ep=[150,50,25,5] # Maximo valor considerado aquel que no supera de media el 25% del tiempo total
    grid_test_freq=[5,1] # Frecuencias optimas observadas despues de los tunings

    all_process_ids=['PPO_Ant_seed1','PPO_Ant_seed2','PPO_Ant_seed3','PPO_Ant_seed4',
                    'PPO_Humanoid_seed1','PPO_Humanoid_seed2','PPO_Humanoid_seed3','PPO_Humanoid_seed4',
                    'PPO_HumanoidStandup_seed1']

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
# Todos son experimentos posteriores a MAEB (entre septiembre y octubre de 2025)
# (esta experimentacion esta resumida en experiments_intuition/README_new/main.pdf )
#==================================================================================================

#--------------------------------------------------------------------------------------------------
# Repetir experimentos ya realizados con nuevos entornos
#--------------------------------------------------------------------------------------------------
if new_experiments:
    #---------------- Comparacion de criterios por entorno
    # global_deg_metric='relative_worsening_to_improvement'
    # local_deg_metric='relative_reward_diff'

    # global_deg_metric='worsening_to_improvement'
    # local_deg_metric='relative_reward_diff'

    global_deg_metric='best_last_deg'
    local_deg_metric='paired_diff_probpos'

    analyzer=ProcessIndependentAnalyzer(306,grid_test_freq=test_freq,grid_test_n_ep=test_n_ep,grid_train_n_ep=train_n_ep,
                                        global_deg_metric=global_deg_metric,local_deg_metric=local_deg_metric)

    analyzer.MAEB_graph_best_criteria_by_time('HalfCheetah',halfcheetah_process_ids,train_n_ep,test_n_ep,test_freq,global_deg_metric,local_deg_metric)
    analyzer.MAEB_graph_best_criteria_by_time('Walker2d',walker2d_process_ids,train_n_ep,test_n_ep,test_freq,global_deg_metric,local_deg_metric)
    analyzer.MAEB_graph_best_criteria_by_time('Ant',ant_process_ids,train_n_ep,test_n_ep,test_freq,global_deg_metric,local_deg_metric)
    analyzer.MAEB_graph_best_criteria_by_time('Humanoid',humanoid_process_ids,train_n_ep,test_n_ep,test_freq,global_deg_metric,local_deg_metric)
    analyzer.MAEB_graph_best_criteria_by_time('HumanoidStandup',humanidstandup_process_ids,train_n_ep,test_n_ep,test_freq,global_deg_metric,local_deg_metric)

    analyzer.graph_best_criteria_by_time_related_deg('HalfCheetah',halfcheetah_process_ids,train_n_ep,test_n_ep,test_freq,global_deg_metric,local_deg_metric)
    analyzer.graph_best_criteria_by_time_related_deg('Walker2d',walker2d_process_ids,train_n_ep,test_n_ep,test_freq,global_deg_metric,local_deg_metric)
    analyzer.graph_best_criteria_by_time_related_deg('Ant',ant_process_ids,train_n_ep,test_n_ep,test_freq,global_deg_metric,local_deg_metric)
    analyzer.graph_best_criteria_by_time_related_deg('Humanoid',humanoid_process_ids,train_n_ep,test_n_ep,test_freq,global_deg_metric,local_deg_metric)
    analyzer.graph_best_criteria_by_time_related_deg('HumanoidStandup',humanidstandup_process_ids,train_n_ep,test_n_ep,test_freq,global_deg_metric,local_deg_metric)

    analyzer.graph_best_criteria_by_time_related_deg('HalfCheetah',halfcheetah_process_ids,train_n_ep,test_n_ep,test_freq,global_deg_metric,local_deg_metric,time_range_type='limit')
    analyzer.graph_best_criteria_by_time_related_deg('Walker2d',walker2d_process_ids,train_n_ep,test_n_ep,test_freq,global_deg_metric,local_deg_metric,time_range_type='limit')
    analyzer.graph_best_criteria_by_time_related_deg('Ant',ant_process_ids,train_n_ep,test_n_ep,test_freq,global_deg_metric,local_deg_metric,time_range_type='limit')
    analyzer.graph_best_criteria_by_time_related_deg('Humanoid',humanoid_process_ids,train_n_ep,test_n_ep,test_freq,global_deg_metric,local_deg_metric,time_range_type='limit')
    analyzer.graph_best_criteria_by_time_related_deg('HumanoidStandup',humanidstandup_process_ids,train_n_ep,test_n_ep,test_freq,global_deg_metric,local_deg_metric,time_range_type='limit')

#--------------------------------------------------------------------------------------------------
# Graficas para conclusion final: cuando usar que criterio en funcion del tiempo disponible
#--------------------------------------------------------------------------------------------------
if experiments_when_which:
    global_deg_metric='best_last_deg'
    local_deg_metric='paired_diff_probpos'

    analyzer=ProcessIndependentAnalyzer(306,grid_test_freq=test_freq,grid_test_n_ep=test_n_ep,grid_train_n_ep=train_n_ep,
                                        global_deg_metric=global_deg_metric,local_deg_metric=local_deg_metric)

    analyzer.graph_when_which_criteria(halfcheetah_process_ids,500,32,1,global_deg_metric,local_deg_metric,'HalfCheetah')
    analyzer.graph_when_which_criteria(ant_process_ids,500,32,1,global_deg_metric,local_deg_metric,'Ant')
    analyzer.graph_when_which_criteria(walker2d_process_ids,500,32,1,global_deg_metric,local_deg_metric,'Walker2d')
    analyzer.graph_when_which_criteria(humanoid_process_ids,500,32,1,global_deg_metric,local_deg_metric,'Humanoid')
    analyzer.graph_when_which_criteria(humanidstandup_process_ids,500,32,1,global_deg_metric,local_deg_metric,'HumanoidStandup')

