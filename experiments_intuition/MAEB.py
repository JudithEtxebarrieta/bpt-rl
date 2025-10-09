'''
Generación de graficas adaptadas para MAEB e ISG charla
'''


from Main import EstimationAnalyzer, CriteriaTuner, ProcessIndependentAnalyzer, EvolutionGrapher

def MAEB_ISG_generate_graphs():

    # Para el analisis individual usaremos el mismo proceso que en la experimentacion preliminar del proyecto
    algo='PPO'
    env='Ant'
    seed=1
    resources='16cpu1gpu_mejorado'
    max_iter=306
    list_n_ep=[500,250,100,50,25,5]
    list_freq=[100,50,25,10,5,1]

    analyzer=EstimationAnalyzer(algo,env,seed,resources,list_n_ep,list_n_ep,max_iter)
    analyzer.MAEB_graph_cost_analysis()#ISG
    analyzer.MAEB_graph_train_vs_test_estimates()#MAEB+ISG

    tuner=CriteriaTuner(algo,env,seed,resources,[500,100,5],list_freq,max_iter)
    tuner.MAEB_graph_best_val_tuning()# MAEB+ISG

    # MAEB+ISG analisis conjunto y ISG definicion degradacion
    process_ids= ['PPO_Ant_seed'+str(seed) for seed in range(1,11)]+['PPO_Humanoid_seed'+str(seed) for seed in range(1,11)]+['PPO_HumanoidStandup_seed'+str(seed) for seed in range(1,11)]
    title='all_Ant_Humanoid_HumanoidStandup'
    global_deg_metric='best_last_deg'
    local_deg_metric='paired_diff_probpos'
    grid_train_n_ep=[500]
    grid_test_n_ep=[5]
    grid_test_freq=[1]

    analyzer=ProcessIndependentAnalyzer(306,
                                global_deg_metric=global_deg_metric,local_deg_metric=local_deg_metric,
                                all_possible_conf=False,
                                grid_train_n_ep=grid_train_n_ep,#list(set([500,250,100,50,25,5]+grid_train_n_ep)),
                                grid_test_n_ep=grid_test_n_ep,#list(set([500,250,100,50,25,5]+grid_test_n_ep)),
                                grid_test_freq=grid_test_freq)#list(set([100,50,25,10,5,1]+grid_test_freq)))
    
    analyzer.MAEB_graph_best_criteria_by_time(title,process_ids,grid_train_n_ep,grid_test_n_ep,grid_test_freq,global_deg_metric,local_deg_metric,MAEB=True)

    # MAEB definicion de degradación (dibujar en la misma curva varias degradaciones, para que se entienda que estamos monitorizando la degradacion con la verdad absoluta)
    grapher=EvolutionGrapher('PPO','Humanoid',6,'16cpu1gpu_mejorado',306)
    grapher.MAEB_graph_degradation_evolution('best_last_deg','paired_diff_probpos',47)

    # Graficas de analisis comparativo extra (con mas detalle)
    ant_process_ids=['PPO_Ant_seed'+str(seed) for seed in range(1,11)]
    humanoid_process_ids=['PPO_Humanoid_seed'+str(seed) for seed in range(1,11)]
    humanidstandup_process_ids=['PPO_HumanoidStandup_seed'+str(seed) for seed in range(1,11)]
    all_process_ids= ['PPO_Ant_seed'+str(seed) for seed in range(1,11)]+['PPO_Humanoid_seed'+str(seed) for seed in range(1,11)]+['PPO_HumanoidStandup_seed'+str(seed) for seed in range(1,11)]

    analyzer.MAEB2_graph_best_criteria_by_time('all_Ant',ant_process_ids,grid_train_n_ep,grid_test_n_ep,grid_test_freq,global_deg_metric,local_deg_metric)
    analyzer.MAEB2_graph_best_criteria_by_time('all_Humanoid',humanoid_process_ids,grid_train_n_ep,grid_test_n_ep,grid_test_freq,global_deg_metric,local_deg_metric)
    analyzer.MAEB2_graph_best_criteria_by_time('all_HumanoidStandup',humanidstandup_process_ids,grid_train_n_ep,grid_test_n_ep,grid_test_freq,global_deg_metric,local_deg_metric)
    analyzer.MAEB2_graph_best_criteria_by_time(title,all_process_ids,grid_train_n_ep,grid_test_n_ep,grid_test_freq,global_deg_metric,local_deg_metric)


MAEB_ISG_generate_graphs()
