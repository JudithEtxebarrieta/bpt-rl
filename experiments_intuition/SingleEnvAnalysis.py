from Main import *

# EXPERIMENTS
experiment_getstarting=False
experiment_estimates_together=False
experiment_per_pack=True

# Todos son experimentos posteriores a MAEB
#--------------------------------------------------------------------------------------------------
# Getstarting analisis por environment
#--------------------------------------------------------------------------------------------------
if experiment_getstarting:
    global_deg_metric='best_last_deg'
    local_deg_metric='paired_diff_probpos'
    analyzer=ProcessIndependentAnalyzer(306,grid_train_n_ep=[500,250,100,50,25,5],grid_test_n_ep=[500,250,100,50,25,5],global_deg_metric=global_deg_metric,local_deg_metric=local_deg_metric)
                             
    analyzer.graph_getstarting_by_env('PPO','HalfCheetah')
    analyzer.graph_getstarting_by_env('PPO','Walker2d')
    analyzer.graph_getstarting_by_env('PPO','Ant')
    analyzer.graph_getstarting_by_env('PPO','Humanoid')
    analyzer.graph_getstarting_by_env('PPO','HumanoidStandup')

    analyzer.graph_getstarting_by_env('PPO','HalfCheetah',prec_with='paired_diff_probpos')
    analyzer.graph_getstarting_by_env('PPO','Walker2d',prec_with='paired_diff_probpos')
    analyzer.graph_getstarting_by_env('PPO','Ant',prec_with='paired_diff_probpos')
    analyzer.graph_getstarting_by_env('PPO','Humanoid',prec_with='paired_diff_probpos')
    analyzer.graph_getstarting_by_env('PPO','HumanoidStandup',prec_with='paired_diff_probpos')

    analyzer.graph_getstarting_by_env('PPO','HalfCheetah',conv_range_type='limit')
    analyzer.graph_getstarting_by_env('PPO','Walker2d',conv_range_type='limit')
    analyzer.graph_getstarting_by_env('PPO','Ant',conv_range_type='limit')
    analyzer.graph_getstarting_by_env('PPO','Humanoid',conv_range_type='limit')
    analyzer.graph_getstarting_by_env('PPO','HumanoidStandup',conv_range_type='limit')


#--------------------------------------------------------------------------------------------------
# Estimaciones truth vs train vs test (peticion de Aritz)
#--------------------------------------------------------------------------------------------------
if experiment_estimates_together:
    envs=['HalfCheetah','Walker2d','Ant','Humanoid','HumanoidStandup']
    seeds=list(range(1,11))

    analyzer=ProcessIndependentAnalyzer(306)
    # analyzer.graph_truth_train_test_together('PPO',envs,seeds,10)
    # analyzer.graph_truth_train_test_together('PPO',envs,seeds,10,opt_conf_per_env=True)
    # analyzer.graph_truth_train_test_together('PPO',envs,seeds,10,together_what='truth')
    # analyzer.graph_truth_train_test_together('PPO',envs,seeds,10,together_what='truth',opt_conf_per_env=True)

    analyzer.graph_truth_train_test_together('PPO',envs,seeds,10,together_what='truth',limit_metric='from_mean')

#--------------------------------------------------------------------------------------------------
# NUEVO: experimento por pack
#--------------------------------------------------------------------------------------------------
if experiment_per_pack:

    seeds=list(range(1,17))+[18,19,20,23]
    analyzer=ProcessIndependentAnalyzer(114)

    # Analizando degradacion y regiones de aprendizaje por proceso
    analyzer.graph_pack_all_truth_with_regions('pack_PPO_BipedalWalker',seeds)
    analyzer.graph_pack_all_truth_with_regions('pack_PPO_BipedalWalker',seeds,
                                               global_deg_metric='best_last_deg',local_deg_metric='paired_diff_probpos')
    analyzer.graph_pack_all_truth_with_regions('pack_PPO_BipedalWalker',seeds,
                                               global_deg_metric='best_last_deg',local_deg_metric='paired_diff_KDEprobpos')
    analyzer.graph_pack_all_truth_with_regions('pack_PPO_BipedalWalker',seeds,
                                               global_deg_metric='norm_from_mean_worsening_to_improvement')

    # Analizando curvas de aprendizaje con diferentes criterios y configuraciones de estos
    analyzer.graph_pack_learning_curves_with_criteria('pack_PPO_BipedalWalker',seeds,default_conf=[5,5,5],cost_perc=0.1) #default_test_n_ep=eval_freq/n_steps
    analyzer.graph_pack_learning_curves_with_criteria('pack_PPO_BipedalWalker',seeds,default_conf=[5,5,5],cost_perc=0.2)
    analyzer.graph_pack_learning_curves_with_criteria('pack_PPO_BipedalWalker',seeds,default_conf=[5,5,5],cost_perc=0.25)

    # Generar datos de precision de seleccion para los diferentes criterios
    analyzer.generate_data_pack_complete_analysis('pack_PPO_BipedalWalker',seeds,cost_perc=0.25)
    analyzer.generate_data_pack_complete_analysis('pack_PPO_BipedalWalker',seeds,
                                                  prec_metric='paired_diff_KDEprob0')