from Main import *

# EXPERIMENTS
experiment_getstarting=False
experiment_estimates_together=False

# Todos son experimentos posteriores a MAEB
#--------------------------------------------------------------------------------------------------
# Getstarting analisis por environment
#--------------------------------------------------------------------------------------------------
if experiment_getstarting:
    global_deg_metric='best_last_deg'
    local_deg_metric='paired_diff_probpos'
    analyzer=ProcessIndependentAnalyzer(306,grid_test_n_ep=[500,250,100,50,25,5],global_deg_metric=global_deg_metric,local_deg_metric=local_deg_metric)
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

#--------------------------------------------------------------------------------------------------
# Estimaciones truth vs train vs test (peticion de Aritz)
#--------------------------------------------------------------------------------------------------
if experiment_estimates_together:
    envs=['HalfCheetah','Walker2d','Ant','Humanoid','HumanoidStandup']
    seeds=list(range(1,11))

    analyzer=ProcessIndependentAnalyzer(306)
    #analyzer.graph_truth_train_test_estimates_together('PPO',envs,seeds,10)
    analyzer.graph_truth_train_test_estimates_together('PPO',envs,seeds,10,estimates_conv='conv')
    #analyzer.graph_truth_train_test_estimates_together('PPO',envs,seeds,10,opt_conf_per_env=True)