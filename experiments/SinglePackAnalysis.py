from Main import *

class ExecutePackAnalysis():

    def __init__(self,library,pack,seeds,
                 train_default_n_ep,test_default_n_ep,test_default_freq,list_freq,
                 global_deg_metric='norm_from_mean_worsening_to_improvement',local_deg_metric='reward_diff',
                 prec_metric='relative_perc_criteria_best',
                 limit_metric='from_first_last'
                 ):
        
        # ----------------- Generar datos necesarios
        datagenerator=DataGenerator(library,pack,seeds,list_freq)

        # ----------------- Graficas principales
        grapher=Grapher(library,pack.replace('pack_',''))

        # Analisis 1
        grapher.graph_deg_criteria_conf_by_regions(pack,100,5,5,global_deg_metric=global_deg_metric,local_deg_metric=local_deg_metric)
        n_ep_train,n_ep_test,freq_test=grapher.graph_deg_criteria_conf_by_regions(pack,100,5,5,global_deg_metric=global_deg_metric,local_deg_metric=local_deg_metric,n_ep_type='with_cost')
        
        n_ep_test,freq_test=0.15,1 # TODO: LunarLanderContinuos deberia salir asi automatico
        # Analisis 2
        datagenerator.add_test_cost_truth(n_ep_test,freq_test)
        grapher.graph_criteria_comparison_by_regions(pack,global_deg_metric=global_deg_metric,local_deg_metric=local_deg_metric,
                                                    prec_metric=prec_metric,limit_metric=limit_metric,
                                                    train_conf=[n_ep_train if n_ep_train!=None else train_default_n_ep ][0],test_conf=str(n_ep_test)+'cost_'+str(freq_test))
        # Recomendacion
        grapher.graph_pack_learning_curves_with_criteria(pack,default_conf=[train_default_n_ep,test_default_n_ep,test_default_freq],optimal_conf=str(n_ep_test)+'cost_'+str(freq_test)) #default_test_n_ep=eval_freq/n_steps
        # grapher.graph_pack_learning_curves_with_criteria(pack,default_conf=[train_default_n_ep,test_default_n_ep,test_default_freq],optimal_conf=n_ep_test+'cost_'+freq_test,curves='estimate_truth')

        grapher.graph_pack_learning_curves_error(pack,default_conf=[train_default_n_ep,test_default_n_ep,test_default_freq],diff='truth',optimal_conf=str(n_ep_test)+'cost_'+str(freq_test))
        # grapher.graph_pack_learning_curves_error(pack,default_conf=[train_default_n_ep,test_default_n_ep,test_default_freq],optimal_conf=n_ep_test+'cost_'+freq_test)

        # ----------------- Graficas secundarias
        # Para mostrar como la degradacion y los limites definen lo que decimos
        grapher.graph_pack_all_truth_with_regions(pack,seeds,
                                                global_deg_metric='norm_from_mean_worsening_to_improvement')
        # grapher.graph_pack_all_truth_with_regions(pack,seeds,
        #                                         global_deg_metric='best_last_deg',local_deg_metric='paired_diff_probpos')
        # grapher.graph_pack_all_truth_with_regions(pack,seeds,
        #                                         global_deg_metric='best_last_deg',local_deg_metric='greater_prob')

        # Para mostrar la estabilidad de la estimacion considerada para truth
        grapher.graph_pack_all_stability_truth_estimator(pack,seeds)
        grapher.graph_pack_all_stability_truth_estimator(pack,seeds,stability_metric='mean_diff')
        grapher.graph_pack_all_stability_truth_estimator(pack,seeds,stability_metric='CI_width')


ExecutePackAnalysis('SB3','pack_PPO_BipedalWalker',list(range(1,17))+[18,19,20,23], #en el cluster estas 20 semilla se ejecutaron antes
                    100,5,5,[50,25,10,5,2,1])

ExecutePackAnalysis('SB3','pack_PPO_LunarLanderContinuous',list(range(3,22))+[1],#la ejecucion de la semilla 3 se ha interrumpido en el cluster y no estan los datos completos
                    100,5,10,[40,20,10,5,2,1])
#TODO: al calcular automaticamente el n_train_ep en n_ep_train,n_ep_test,freq_test=grapher.graph_deg_criteria_conf_by_regions,
# veo dos cosas raras: 1) lo que veo en la grafica que deberia ser la configuracion optima (primera vez que los intervalos dentro de franja)
# no coincide con la grafica; 2) si quiero ejecutar la grafica siguiente  grapher.graph_criteria_comparison_by_regions con n_train_ep=250, o 500
# me da un error. Mirar porque pasan esas cosas. Puede ser que al generar los datos mientras corregia errores algo se haya guardado mal?



