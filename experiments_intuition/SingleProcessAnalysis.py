import os
import pandas as pd
from Main import EvolutionGrapher, EstimationAnalyzer, CriteriaTuner, ProcessIndependentAnalysis

# EXPERIMENTOS
experiments_intuition=False
experiments_more_env=False
experiments_invest_time_evol=False

#==================================================================================================
# Experimentos realizados para ganar intuicion 
# (esta experimentación esta resumida cronologicamente en experiments_intuition/README/main.pdf )
#==================================================================================================

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
        grapher.learning_curve()
        grapher.graph_degradation_evolution('mean_update_deg','greater_prob')
        grapher.graph_degradation_evolution('weighted_mean_best_later_deg','greater_prob')
        grapher.graph_degradation_evolution('weighted_mean_best_later_deg','paired_diff_probpos_meanpos')
        grapher.graph_degradation_evolution('best_last_deg','greater_prob')
        grapher.graph_degradation_evolution('best_last_deg','paired_diff_median')
        grapher.graph_degradation_evolution('best_last_deg','paired_diff_probpos')
        # grapher.graph_degradation_evolution('relative_worsening_to_improvement','relative_reward_diff')
        # grapher.graph_degradation_evolution('worsening_to_improvement','relative_reward_diff')
        # grapher.graph_degradation_evolution('best_last_deg','paired_diff_probpos')
        # grapher.graph_degradation_evolution('relative_best_last_deg','paired_diff_probpos')
    
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

#==================================================================================================
# Experimentos posteriores
#==================================================================================================

if experiments_more_env:
    
    #------------- Mas environments
    for i in range(1,11):
        SingleProcessAnalisys('PPO','HalfCheetah',i,'16cpu1gpu_mejorado',306)
        SingleProcessAnalisys('PPO','Walker2d',i,'16cpu1gpu_mejorado',306)
        SingleProcessAnalisys('PPO','Ant',i,'16cpu1gpu_mejorado',306)
        SingleProcessAnalisys('PPO','Humanoid',i,'16cpu1gpu_mejorado',306)
        SingleProcessAnalisys('PPO','HumanoidStandup',i,'16cpu1gpu_mejorado',306)

    #------------- Configuraciones optimas por entorno
    df=pd.read_csv('experiments_intuition/results/CriteriaComparison/data/criteria_conf_by_process.csv')
    df_opt_conf_per_env=[]
    for env in ['HalfCheetah','Walker2d','Ant','Humanoid','HumanoidStandup']:
        # Filas asociadas al env
        df_env=df[df["process_id"].str.contains(env+'_')]
        opt_train_n_ep=max(set(df_env['train_n_ep']), key=list(df_env['train_n_ep']).count)
        opt_test_n_ep,opt_test_freq = df_env.value_counts(subset=["test_n_ep", "test_freq"]).idxmax()
        df_opt_conf_per_env.append(['PPO',env,opt_train_n_ep,opt_test_n_ep,opt_test_freq])
    # Todos los entornos
    opt_train_n_ep=max(set(df['train_n_ep']), key=list(df_env['train_n_ep']).count)
    opt_test_n_ep,opt_test_freq = df.value_counts(subset=["test_n_ep", "test_freq"]).idxmax()
    df_opt_conf_per_env.append(['PPO','All',opt_train_n_ep,opt_test_n_ep,opt_test_freq])

    df_opt_conf_per_env=pd.DataFrame(df_opt_conf_per_env,columns=['algo','env','train_n_ep','test_n_ep','test_freq'])
    df_opt_conf_per_env.to_csv("experiments_intuition/results/CriteriaComparison/data/opt_criteria_conf_by_env.csv", index=False)
    


if experiments_invest_time_evol:

    analyzer=EstimationAnalyzer('PPO','HalfCheetah',1,'16cpu1gpu_mejorado',[],[16,10],306)
    analyzer.graph_invest_time_evolution([500,250,100,50,25,16,10,5])
    analyzer=EstimationAnalyzer('PPO','Walker2d',1,'16cpu1gpu_mejorado',[],[16,10],306)
    analyzer.graph_invest_time_evolution([500,250,100,50,25,16,10,5])