from Main import *

library='SB3'
data_path='experiments/results/data'
figure_path='experiments/results/figures'
seeds=list(range(1,31))
all_packs=[ # Box2D
            'pack_PPO_LunarLanderContinuous',
            'pack_PPO_BipedalWalker',
            # MuJoCo
            'pack_PPO_Swimmer',
            'pack_PPO_HalfCheetah',
            'pack_PPO_Hopper',
            'pack_PPO_Ant',
            'pack_PPO_Walker2d'            
                            ]

global_deg_metric='norm_from_mean_worsening_to_improvement'
local_deg_metric='reward_diff'
prec_metric='relative_perc_criteria_best'
eff_metric='first_time_to_same_reward'

class Grapher():

    def __init__(self,library,all_packs,seeds,data_path,figure_path):
        self.library=library
        self.all_packs=all_packs
        self.seeds=seeds
        self.data_path=data_path
        self.figure_path=figure_path


    def degradation(self):

        def plot_pack_degradation(ax,deg_list,pack_name='',title='',not_last_pack=False):

            data = np.array(deg_list)

            bins = 10
            bin_edges = np.linspace(0, 1, bins + 1)
            counts, _ = np.histogram(data, bins=bin_edges)
            
            percentages = counts / counts.sum() 

            # Escala de gris
            colors = [str(1 - p) for p in percentages]

            bin_width = bin_edges[1] - bin_edges[0]

            # Dibujar barras todas de la misma altura (e.g, 1)
            for left, color in zip(bin_edges[:-1], colors):
                ax.bar(left, 1, width=bin_width, color=color, edgecolor=None, align='edge')

            # Linea de la mediana
            ax.axvline(np.median(data), color='red',linewidth=2)

            ax.set_xlim(-0.05, 1.05)
            ax.set_title(title)
            ax.set_ylabel(pack_name,rotation=0, labelpad=80)
            if not_last_pack:   
                ax.set_xticks([])
            ax.set_yticks([]) 

        fig, axs = plt.subplots(len(self.all_packs),3, figsize=(10,0.5*len(self.all_packs)))
        plt.subplots_adjust(top=0.9,bottom=0.1,left=0.2,right=0.98, hspace=0.3,wspace=0.03)

        for i,pack in enumerate(all_packs):

            pack_path=self.data_path+'/'+pack.replace('pack',self.library)+'/'

            # Obtener los datos de degradacion por region de este pack
            deg1,deg2,deg3=DataConverter.from_df_data_to_graph_data(
                                    [pack_path+'learning_regions.csv',pack_path+'deg_evolution.csv'],pack,'deg_distribution',
                                    global_deg_metric=global_deg_metric,local_deg_metric=local_deg_metric
                                ) 

            # Dibujar distribucion de degradacion de las tres regiones
            title=[['Initialization','Learning','Stabilization']if i==0 else ['']*3][0]
            not_last_pack=[False if i==len(all_packs)-1 else True][0]
            plot_pack_degradation(axs[i,0],deg1,pack_name=pack.replace('pack_PPO_',''),title=title[0],not_last_pack=not_last_pack)
            plot_pack_degradation(axs[i,1],deg2,title=title[1],not_last_pack=not_last_pack)
            plot_pack_degradation(axs[i,2],deg3,title=title[2],not_last_pack=not_last_pack)
        
        plt.savefig(self.figure_path+'/all_setups/degradation.pdf')

    def criteria_precision_or_efficiency(self,prec_or_eff='prec'):

        def plot_pack_criteria_prec_or_eff(ax,listas,pack_name='',title='',last_pack=False):
            colors=['green','orange','blue']
            color_marker_map = {'green': 's','orange': '^','blue': 'o'}
            for y, data,color in zip(range(len(listas)), listas,colors):

                ax.hlines(y, np.percentile(data, 25), np.percentile(data, 75), color=color,linewidth=1)
                ax.vlines(np.percentile(data, 25), y - 0.2, y + 0.2, color=color)
                ax.vlines(np.percentile(data, 75), y - 0.2, y + 0.2, color=color)
                ax.plot(np.median(data), y, color_marker_map[color], color=color,markersize=4)

            ax.grid(axis='x', linestyle='--',alpha=0.4)
            ax.set_title(title)
            ax.set_xlim(-0.05,1.05)
            ax.set_ylabel(pack_name,rotation=0, labelpad=70,fontsize=8)
            ax.set_yticks([])
            ax.tick_params(axis='x', labelsize=8)
            if not last_pack:
                ax.tick_params(axis='x', labelbottom=False)
            else:
                if pack_name!='':
            
                    legend_elements = [
                        Line2D([0], [0], marker='o', color='blue', label='Last', markersize=6, linestyle=''),
                        Line2D([0], [0], marker='^', color='orange', label='Train', markersize=6, linestyle=''),
                        Line2D([0], [0], marker='s', color='green', label='Test', markersize=6, linestyle='')
                    ]
                    ax.legend(handles=legend_elements, loc='upper center',bbox_to_anchor=(0.5, -0.4), ncol=3, frameon=False)
            

        fig, axs = plt.subplots(len(self.all_packs),3, figsize=(8.5,0.6*len(self.all_packs)))
        plt.subplots_adjust(top=0.9,bottom=0.2,left=0.2,right=0.98, hspace=0.1,wspace=0.03)
        
        for ax in axs.flat:
            for spine in ax.spines.values():
                spine.set_linewidth(0.05)

        df_conf=pd.read_csv(self.data_path+'/configurations.csv')
        train_conf=str(int(df_conf.loc[df_conf["pack"] == 'all', "train_opt"].iloc[0]))
        conf_str=df_conf.loc[df_conf["pack"] == 'all', "test_cost_freq_opt"].iloc[0]
        test_conf=conf_str.split('_')[0]+'_'+conf_str.split('_')[1]+'cost'

        for i,pack in enumerate(all_packs):

            pack_path=self.data_path+'/'+pack.replace('pack',self.library)+'/'

            # Obtener los datos de prec/eff por region de este pack
            last1,last2,last3=DataConverter.from_df_data_to_graph_data(
                            [pack_path+'learning_regions.csv',pack_path+'df_last_'+prec_or_eff+'.csv'],pack,which_graph='last_'+prec_or_eff,
                            prec_metric=prec_metric,eff_metric=eff_metric
                        )
            train1,train2,train3=DataConverter.from_df_data_to_graph_data(
                            [pack_path+'learning_regions.csv',pack_path+'df_train_'+prec_or_eff+'.csv'],pack,which_graph='train_'+prec_or_eff,
                            prec_metric=prec_metric,eff_metric=eff_metric,train_conf=train_conf
                        )
            test1,test2,test3=DataConverter.from_df_data_to_graph_data(
                            [pack_path+'learning_regions.csv',pack_path+'df_test_'+prec_or_eff+'.csv'],pack,which_graph='test_'+prec_or_eff,
                            prec_metric=prec_metric,eff_metric=eff_metric,test_conf=test_conf
                        )

            # Dibujar distribucion de prec/eff de las tres regiones
            title=[['Initialization','Learning','Stabilization']if i==0 else ['']*3][0]
            last_pack=[True if i==len(all_packs)-1 else False][0]
            plot_pack_criteria_prec_or_eff(axs[i,0],[last1,train1,test1][::-1],pack_name=pack.replace('pack_PPO_',''),title=title[0],last_pack=last_pack)
            plot_pack_criteria_prec_or_eff(axs[i,1],[last2,train2,test2][::-1],title=title[1],last_pack=last_pack)
            plot_pack_criteria_prec_or_eff(axs[i,2],[last3,train3,test3][::-1],title=title[2],last_pack=last_pack)
        
        plt.savefig(self.figure_path+'/all_setups/criteria_'+prec_or_eff+'.pdf')
        
    def criteria_comparison(self,comparison_type='how_times_best'):
        '''
        `comparison_type`: puede ser 'how_times_best', 'in_which_deg_best', 'with_what_prec_diff_best'
        '''

        df_conf=pd.read_csv(self.data_path+'/configurations.csv')
        train_conf=str(int(df_conf.loc[df_conf["pack"] == 'all', "train_opt"].iloc[0]))
        conf_str=df_conf.loc[df_conf["pack"] == 'all', "test_cost_freq_opt"].iloc[0]
        test_conf=conf_str.split('_')[0]+'_'+conf_str.split('_')[1]+'cost'

        legend_elements = [
                Line2D([0], [0], marker='o', color='black', label='Last', markersize=6, linestyle=''),
                Line2D([0], [0], marker='^', color='black', label='Train', markersize=6, linestyle=''),
                Line2D([0], [0], marker='s', color='black', label='Test', markersize=6, linestyle='')
            ]

        # Funciones para plotear datos por pack-region
        def plot_how_many_times_better(ax,data,pack_name='',title='',not_last_pack=True):

            color_marker_map = {'green': 's','orange': '^','blue': 'o'}
            colors = [['blue', 'orange'],['blue', 'green'],['orange', 'green']]

            for i, ((left_val, right_val), (color_left, color_right)) in enumerate(zip(data, colors)):
                
                ax.barh(y=i,width=left_val,left=0,color=color_left,height=1)
                ax.barh(y=i,width=right_val,left=1 - right_val,color=color_right,height=1)

                marker_left = color_marker_map.get(color_left, 'o')
                marker_right = color_marker_map.get(color_right, 'o')  
                if left_val>0:       
                    ax.plot(0, i,marker=marker_left,color='black',markersize=3) # Punta izquierda
                if right_val>0:
                    ax.plot(1 , i,marker=marker_right,color='black',markersize=3) # Punta derecha

            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.5, len(data)-0.5)
            ax.set_yticks([0, 1, 2])
            ax.set_ylabel(pack_name,rotation=0, labelpad=70)
            ax.tick_params(axis='y', left=False, labelleft=False)
            if not_last_pack:
                ax.set_xticklabels([])
                if pack_name!='':
                    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -1.6), ncol=3, frameon=False)

            ax.set_title(title)
        
        def plot_in_which_deg_best(ax,data,pack_name='',title='',not_last_pack=True):

            colors = [['blue', 'orange'],['blue', 'green'],['orange', 'green']]
            color_marker_map = {'blue': 'o','orange': '^','green': 's'}

            y_spacing = 0.5
            hist_height = 0.4
            bins = 10

            for i, (pair, pair_colors) in enumerate(zip(data, colors)):

                
                bar_height = hist_height / len(pair_colors) # Para que las barras de un mismo bin no se solapen, dividimos verticalmente

                for j, (sublist, color) in enumerate(zip(pair, pair_colors)):
                    if len(sublist)>0:
                        sublist = np.asarray(sublist)

                        # Histograma normalizado a porcentaje
                        hist, bin_edges = np.histogram(sublist, bins=bins, range=(0,1), density=False)
                        percentages = hist / hist.sum()  # entre 0 y 1

                        bin_width = bin_edges[1] - bin_edges[0]

                        # Color proporcional al porcentaje 
                        base_rgb = np.array(mcolors.to_rgb(color))
                        colors_scaled = [tuple(base_rgb * p + (1-p)) for p in percentages]  # mezcla con blanco

                        # Dibujar barras
                        for left, col in zip(bin_edges[:-1], colors_scaled):
                            ax.bar(left, bar_height, width=bin_width,
                                bottom=i*y_spacing + j*bar_height,
                                align='edge', color=col, edgecolor=None)

                        # Mediana 
                        median_val = np.median(sublist)
                        bottom_y = i*y_spacing + j*bar_height
                        mid_y = bottom_y + bar_height/2
                        marker = color_marker_map.get(color, 'o')
                        ax.plot(median_val, mid_y, marker=marker, color='black', markersize=3)


            
            ax.set_yticks([i*y_spacing for i in range(len(data))])
            ax.tick_params(axis='y', left=False, labelleft=False)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylabel(pack_name,rotation=0, labelpad=70)
            if not_last_pack:
                ax.set_xticklabels([])
                if pack_name!='':
                    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -1.6), ncol=3, frameon=False)

            ax.set_title(title)

        def plot_with_what_prec_diff_best(ax,data,pack_name='',title='',not_last_pack=True):
            colors = [
                        [["#0000FF", "#FADDBB"], ["#FF8800", "#BABAFF"]],
                        [["#0000FF", "#C7EBC7"], ["green", "#BABAFF"]],
                        [["#FF8800", "#C7EBC7"], ["green", "#FADDBB"]]
                    ]
            color_marker_map = {"#0000FF": 'o', "#FF8800": '^', "green": 's',
                                "#BABAFF": 'o', "#FADDBB": '^', "#C7EBC7": 's'
            }
            
            region_spacing = 0.9
            level_spacing = 0.35      # separacion entre subniveles
            inner_offset = 0.06       # separacion pequeña entre los dos segmentos
            cap_height = 0.04         # tamaño de los topes verticales

            # Dibujar los intervalos
            for i, (region, region_colors) in enumerate(zip(data, colors)):
                base_y = i * region_spacing
                for j, (sublist, sub_colors) in enumerate(zip(region, region_colors)):
                    base_level_y = base_y + j * level_spacing
                    for k, (subsubdata, color) in enumerate(zip(sublist, sub_colors)):
                        offset = inner_offset if k == 0 else -inner_offset
                        y_pos = base_level_y + offset
    
                        if len(subsubdata)>0:
                            ax.hlines(y_pos, np.percentile(subsubdata, 25), np.percentile(subsubdata, 75), color=color,linewidth=0.25)
                            ax.vlines(np.percentile(subsubdata, 25),  y_pos - cap_height, y_pos + cap_height, color=color)
                            ax.vlines(np.percentile(subsubdata, 75), y_pos - cap_height, y_pos + cap_height, color=color)
                        else:
                            subsubdata=[2] # para que cuando los dos criterios estan completamente empatados, en la grafica no aparezca ningun CI dibujado pero se mantengan los margenes

                        marker = color_marker_map.get(color, 'o')  # default circulo si color no mapeado
                        ax.plot(np.median(subsubdata), y_pos, marker=marker, color=color, markersize=3)


            for i in range(1, len(data)):
                ax.axhline(i * region_spacing - level_spacing / 2 -0.1, color='gray', linestyle='-',linewidth=0.5,alpha=0.5)
                

            region_centers = [i * region_spacing + level_spacing / 2 for i in range(len(data))]
            ax.set_yticks(region_centers)
            ax.set_ylabel(pack_name,rotation=0, labelpad=70)
            ax.tick_params(axis='y', left=False, labelleft=False)
            ax.set_xlim(-0.05,1.05)
            ax.grid(axis='x', linestyle='--', color='gray', linewidth=0.6,alpha=0.5)

            if not_last_pack:
                ax.set_xticklabels([])
                if pack_name!='':
                    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -1.6), ncol=3, frameon=False)


            ax.set_title(title)


        # Graficas
        if comparison_type=='with_what_prec_diff_best':
            fig, axs = plt.subplots(len(self.all_packs),3, figsize=(10,1.3*len(self.all_packs)))
            plt.subplots_adjust(top=0.9,bottom=0.1,left=0.2,right=0.98, hspace=0.1,wspace=0.03)
        if comparison_type=='how_times_best':
            fig, axs = plt.subplots(len(self.all_packs),3, figsize=(10,0.6*len(self.all_packs)))
            plt.subplots_adjust(top=0.9,bottom=0.2,left=0.2,right=0.98, hspace=0.1,wspace=0.03)
        if comparison_type=='in_which_deg_best':
            fig, axs = plt.subplots(len(self.all_packs),3, figsize=(10,0.8*len(self.all_packs)))
            plt.subplots_adjust(top=0.9,bottom=0.1,left=0.2,right=0.98, hspace=0.1,wspace=0.03)

        for i,pack in enumerate(all_packs):

            pack_path=self.data_path+'/'+pack.replace('pack',self.library)+'/'
            title=[['Initialization','Learning','Stabilization']if i==0 else ['']*3][0]
            not_last_pack=[False if i==len(all_packs)-1 else True][0]
            
            if comparison_type=='how_times_best':

                # Obtener los datos de la medida de comapracion por region de este pack            
                matrix1,matrix2,matrix3,_,_,_=DataConverter.from_df_data_to_graph_data(
                                        [pack_path+'learning_regions.csv',pack_path+'df_last_truth.csv',
                                        pack_path+'df_train_truth.csv',pack_path+'df_test_truth.csv'],
                                        pack,which_graph=comparison_type,train_conf=train_conf,test_conf=test_conf)
                # Dibujar distribucion de la medida de comparacion de las tres regiones
                plot_how_many_times_better(axs[i,0],matrix1,pack_name=pack.replace('pack_PPO_',''),title=title[0],not_last_pack=not_last_pack)
                plot_how_many_times_better(axs[i,1],matrix2,title=title[1],not_last_pack=not_last_pack)
                plot_how_many_times_better(axs[i,2],matrix3,title=title[2],not_last_pack=not_last_pack)

            if comparison_type=='in_which_deg_best':
                matrix1,matrix2,matrix3=DataConverter.from_df_data_to_graph_data(
                                [pack_path+'learning_regions.csv',pack_path+'df_last_truth.csv',
                                pack_path+'df_train_truth.csv',pack_path+'df_test_truth.csv',pack_path+'deg_evolution.csv'],
                                pack,which_graph=comparison_type,
                                train_conf=train_conf,test_conf=test_conf,
                                global_deg_metric=global_deg_metric,local_deg_metric=local_deg_metric
                                )
                
                plot_in_which_deg_best(axs[i,0],matrix1,pack_name=pack.replace('pack_PPO_',''),title=title[0],not_last_pack=not_last_pack)
                plot_in_which_deg_best(axs[i,1],matrix2,title=title[1],not_last_pack=not_last_pack)
                plot_in_which_deg_best(axs[i,2],matrix3,title=title[2],not_last_pack=not_last_pack)
                
            if comparison_type=='with_what_prec_diff_best':

                matrix1,matrix2,matrix3=DataConverter.from_df_data_to_graph_data(
                                    [pack_path+'learning_regions.csv',pack_path+'df_last_truth.csv',
                                    pack_path+'df_train_truth.csv',pack_path+'df_test_truth.csv',
                                    pack_path+'df_last_prec.csv',pack_path+'df_train_prec.csv',pack_path+'df_test_prec.csv'],pack,
                                    which_graph=comparison_type,train_conf=train_conf,test_conf=test_conf,prec_metric=prec_metric
                                    )

                plot_with_what_prec_diff_best(axs[i,0],matrix1,pack_name=pack.replace('pack_PPO_',''),title=title[0],not_last_pack=not_last_pack)
                plot_with_what_prec_diff_best(axs[i,1],matrix2,title=title[1],not_last_pack=not_last_pack)
                plot_with_what_prec_diff_best(axs[i,2],matrix3,title=title[2],not_last_pack=not_last_pack)

        plt.savefig(self.figure_path+'/all_setups/criteria_comparison_'+comparison_type+'.pdf')
        
    def criteria_consequences(self,consequence_type='learning_curve'):

        df_conf=pd.read_csv(self.data_path+'/configurations.csv')
        conf_str=df_conf.loc[df_conf["pack"] == 'all', "test_cost_freq_opt"].iloc[0]
        test_cost_freq=conf_str.split('_')[0]+'_'+conf_str.split('_')[1]+'cost'

        legend_elements = [
                            Line2D([0], [0], marker='o', color='blue', label='Last',markersize=6, linestyle=''),
                            Line2D([0], [0], marker='^', color='orange', label='Train default',markersize=6, linestyle=''),
                            Line2D([0], [0], marker='D', color="#A52D81", label='Test default',markersize=6, linestyle=''),
                            Line2D([0], [0], marker='s', color='green', label='Test with cost',markersize=6, linestyle='')
                            
                        ]

        def plot_mediana_ci(ax, df, color, marker=None):
            ax.plot(df.index , df.median(axis=1), color=color, marker=marker,markevery=int(len(df.index)/10),linewidth=1)
            ax.fill_between(df.index, df.quantile(0.25, axis=1), df.quantile(0.75, axis=1), color=color, alpha=0.2)
        
        def plot_pack_criteria_learning_curves(axs,pack_name,pack_indx,first_pack=False):

            #---- Truth vs last
            plot_mediana_ci(axs[0,pack_indx], df_truth_pack, color='black')
            plot_mediana_ci(axs[0,pack_indx], df_last_pack, color='blue', marker='o')

            #---- Truth vs default train
            plot_mediana_ci(axs[1,pack_indx], df_truth_pack, color='black')
            df_train_pack_conf=df_train_pack.loc[:, df_train_pack.columns.str.endswith('_'+str(train_default))]
            plot_mediana_ci(axs[1,pack_indx], df_train_pack_conf, color='orange', marker='^')
            
            #---- Truth vs default test
            plot_mediana_ci(axs[2,pack_indx], df_truth_pack, color='black')
            df_test_pack_conf=df_test_pack.loc[:, df_test_pack.columns.str.endswith('_'+test_default)]
            plot_mediana_ci(axs[2,pack_indx], df_test_pack_conf, color="#A52D81",marker='D')
            
            #---- Truth vs recomended test
            plot_mediana_ci(axs[3,pack_indx], df_truth_pack, color='black')
            df_test_pack_conf=df_test_pack.loc[:, df_test_pack.columns.str.endswith('_'+test_cost_freq)]
            plot_mediana_ci(axs[3,pack_indx], df_test_pack_conf, color="green", marker='s')

            if first_pack:
                axs[0,pack_indx].set_ylabel('Truth of selected as best',fontsize=8)
                axs[1,pack_indx].set_ylabel('Truth of selected as best',fontsize=8)
                axs[2,pack_indx].set_ylabel('Truth of selected as best',fontsize=8)
                axs[3,pack_indx].set_ylabel('Truth of selected as best',fontsize=8)
                if pack_name!='':
                    axs[3,pack_indx].legend(handles=legend_elements,loc='upper center',bbox_to_anchor=(2, -0.3),ncol=4,frameon=False)
            axs[0,pack_indx].set_title(pack_name)
            axs[3,pack_indx].set_xlabel('Number of iterations',fontsize=8)

        def plot_pack_criteria_cummulative_diff(axs,pack_name,pack_indx,first_pack=False):
            #---- Obtener listas con diferencia pareada de evolucion de estimaciones de las politicas seleccionadas
            last_paired_diff,train_paired_diff,test_paired_diff,cost_freq_paired_diff=[],[],[],[]

            for pack_seed in list(df_truth_pack.columns):

                truth=np.array(df_truth_pack[pack_seed].tolist())
                last=np.array(df_last_pack[pack_seed].tolist())
                train=np.array(df_train_pack[pack_seed+'_'+str(train_default)].tolist())
                test=np.array(df_test_pack[pack_seed+'_'+test_default].tolist())
                test_with_cost_freq=np.array(df_test_pack[pack_seed+'_'+test_cost_freq].tolist())

                cost_freq_paired_diff.append(abs(truth-test_with_cost_freq))
                last_paired_diff.append(abs(truth-last))
                train_paired_diff.append(abs(truth-train))
                test_paired_diff.append(abs(truth-test))
                
            #---- Graficas de error acumulado pareado entre curvas de aprendizaje truth vs estimadas
            def plot_cumulative_learning_curve_paired_diff(ax,data,color,nombre=None):
                data = np.array(data)
                accumulated = np.zeros(data.shape[1])
                
                for seed in data:
                    new_accumulated = accumulated + seed
                    ax.fill_between(np.arange(data.shape[1]),accumulated,new_accumulated,color=color,alpha=0.6,edgecolor='none') # area entre la curva anterior y la nueva
                    ax.plot(np.arange(data.shape[1]),new_accumulated,color=color,linewidth=0.8) # Curva superior de esta capa
                    accumulated = new_accumulated

            plot_cumulative_learning_curve_paired_diff(axs[0,pack_indx],last_paired_diff,'blue',nombre=True)
            plot_cumulative_learning_curve_paired_diff(axs[1,pack_indx],train_paired_diff,'orange')
            plot_cumulative_learning_curve_paired_diff(axs[2,pack_indx],test_paired_diff,'#A52D81')
            plot_cumulative_learning_curve_paired_diff(axs[3,pack_indx],cost_freq_paired_diff,"green")

            if first_pack:
                if pack_name!='':
                    axs[3,pack_indx].legend(handles=legend_elements,loc='upper center',bbox_to_anchor=(2, -0.3),ncol=4,frameon=False)
                for i in range(4):
                    axs[i,pack_indx].set_ylabel("Cumulative paired difference\nbetween truth and criteria\nlearning curve",fontsize=8)
            for i in range(4):
                axs[i, pack_indx].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
                if i!=0:
                    axs[i, pack_indx].yaxis.get_offset_text().set_visible(False)

            axs[0,pack_indx].set_title(pack_name)
            axs[3,pack_indx].set_xlabel('Number of iterations',fontsize=8)

        def plot_pack_early_stopping(axs,pack_name,pack_indx,first_pack=False,early_stopping_graph='all'):

            test_n_ep_default,test_freq_default=int(test_default.split('_')[0]),int(test_default.split('_')[1])
            test_n_ep,test_freq=int(conf_str.split('_')[0]),float(conf_str.split('_')[1])

            # Obtener datos para las graficas
            matrix_best=EarlyStopping.successive_halving(self.data_path+'/',pack)
            matrix_last=EarlyStopping.successive_halving(self.data_path+'/',pack,criteria='last')
            matrix_train_default=EarlyStopping.successive_halving(self.data_path+'/',pack,criteria='train',conf=[train_default,None])
            matrix_test_default=EarlyStopping.successive_halving(self.data_path+'/',pack,criteria='test',conf=[test_n_ep_default,test_freq_default])
            matrix_test=EarlyStopping.successive_halving(self.data_path+'/',pack,criteria='test',conf=[test_n_ep,test_freq])

            def plot_succesive_halving(ax,matrix_criteria,color):
                max_long=max([len(i) for i in matrix_criteria])
                for truth_evol in matrix_criteria:
                    ax.plot(range(len(truth_evol)) , truth_evol, color=color,linewidth=1)  
                    if len(truth_evol)!=max_long:
                        ax.axvline(x=len(truth_evol), color='red', linewidth=1)

            if early_stopping_graph=='all':
                plot_succesive_halving(axs[0,pack_indx],matrix_best,'black')
                plot_succesive_halving(axs[1,pack_indx],matrix_last,'blue')
                plot_succesive_halving(axs[2,pack_indx],matrix_train_default,'orange')
                plot_succesive_halving(axs[3,pack_indx],matrix_test_default,'purple')
                plot_succesive_halving(axs[4,pack_indx],matrix_test,'green')

                if first_pack:
                    if pack_name!='':
                        axs[3,pack_indx].legend(handles=legend_elements,loc='upper center',bbox_to_anchor=(2, -1.4),ncol=4,frameon=False)
                    for i in range(4):
                        axs[i,pack_indx].set_ylabel("Truth of selected as best",fontsize=8)

                axs[0,pack_indx].set_title(pack_name)
                axs[4,pack_indx].set_xlabel('Number of iterations',fontsize=8)

            def plot_resource_allocation(ax, listas, colors):

                # 1. rango global
                all_data = [x for lista in listas for sub in lista for x in sub]
                min_val = min(all_data)
                max_val = max(all_data)
                bins = np.linspace(min_val, max_val, 4)
                width = bins[1] - bins[0]
                n_groups = len(listas)

                # espacio interno dentro de cada bin
                group_height = width / (n_groups + 1)

                # 2. calcular histogramas
                histogramas = []
                for lista_de_listas in listas:
                    datos = [x for sub in lista_de_listas for x in sub]
                    counts, _ = np.histogram(datos, bins=bins)
                    histogramas.append(counts)

                # 3. dibujar: bins apilados verticalmente
                for b in range(len(bins) - 1):
                    y_base = bins[b]
                    for i, counts in enumerate(histogramas):
                        y = y_base + (i + 1) * group_height
                        ax.barh(y,counts[b],height=group_height * 0.8,left=0,color=colors[i],align='center')

                # 4. estética
                ax.set_ylim(min_val, max_val)
                ax.set_xlim(0, max(max(h) for h in histogramas) * 1.1)

                bin_centers = bins[:-1] + width / 2
                ax.set_yticks(bin_centers)

                ax.set_xlabel('Number of times')
                if pack_indx==0:
                    ax.set_ylabel('Truth values')
                    ax.set_yticklabels(["low","middle","high"])
                    ax.legend(handles=legend_elements,loc='upper center',bbox_to_anchor=(2, -0.3),ncol=4,frameon=False)

                    for label in ax.get_yticklabels():
                        label.set_rotation(0)
                else:
                    ax.set_yticklabels(["","",""])

            if early_stopping_graph=='summary':
                plot_resource_allocation(axs[pack_indx],
                                            [matrix_best,matrix_last,matrix_train_default,matrix_test_default,matrix_test],
                                            ['black','blue','orange','purple','green'])
                axs[pack_indx].set_title(pack_name)


        if consequence_type in ['learning_curve','cummulative_diff' ]:
            fig, axs = plt.subplots(4,len(self.all_packs), figsize=(3*len(self.all_packs),8),sharex='col',sharey='col')
            plt.subplots_adjust(top=0.9,bottom=0.1,left=0.2,right=0.98, hspace=0.05,wspace=0.3)
        if consequence_type=='early_stopping_all':
            fig, axs = plt.subplots(5,len(self.all_packs), figsize=(3*len(self.all_packs),10),sharex='col',sharey='col')
            plt.subplots_adjust(top=0.9,bottom=0.1,left=0.2,right=0.98, hspace=0.05,wspace=0.3)
        if consequence_type=='early_stopping_summary':
            fig, axs = plt.subplots(1,len(self.all_packs), figsize=(3*len(self.all_packs),3))
            plt.subplots_adjust(top=0.9,bottom=0.3,left=0.2,right=0.98, hspace=0.05,wspace=0.1)

        for i,pack in enumerate(all_packs):

            pack_path=self.data_path+'/'+pack.replace('pack',self.library)+'/'
            first_pack=[True if i==0 else False][0]

            # Leer bases de datos ya generadas y reducirlas al pack de interes
            df_truth_pack=pd.read_csv(pack_path+'df_best_truth.csv').filter(like=pack)
            df_train_pack=pd.read_csv(pack_path+'df_train_truth.csv').filter(like=pack)
            df_test_pack=pd.read_csv(pack_path+'df_test_truth.csv').filter(like=pack)
            df_last_pack=pd.read_csv(pack_path+'df_last_truth.csv').filter(like=pack)

            train_default=int(df_conf.loc[df_conf["pack"] == pack, "train_default"].iloc[0])
            test_default=df_conf.loc[df_conf["pack"] == pack, "test_default"].iloc[0]

            # Dibujar las 4 curvas de aprendizaje
            if consequence_type=='learning_curve':
                plot_pack_criteria_learning_curves(axs,pack.replace('pack_PPO_',''),i,first_pack=first_pack)
            if consequence_type=='cummulative_diff':
                plot_pack_criteria_cummulative_diff(axs,pack.replace('pack_PPO_',''),i,first_pack=first_pack)
            if consequence_type=='early_stopping_all':
                plot_pack_early_stopping(axs,pack.replace('pack_PPO_',''),i,first_pack=first_pack)
            if consequence_type=='early_stopping_summary':
                plot_pack_early_stopping(axs,pack.replace('pack_PPO_',''),i,first_pack=first_pack,early_stopping_graph='summary')
            

        plt.savefig(self.figure_path+'/all_setups/criteria_consequences_'+consequence_type+'.pdf')


grapher=Grapher(library,all_packs,seeds,data_path,figure_path) 
grapher.degradation()
grapher.criteria_precision_or_efficiency()
grapher.criteria_precision_or_efficiency(prec_or_eff='eff')
grapher.criteria_comparison()
grapher.criteria_comparison(comparison_type='in_which_deg_best')
grapher.criteria_comparison(comparison_type='with_what_prec_diff_best')
grapher.criteria_consequences()
grapher.criteria_consequences(consequence_type='cummulative_diff')
grapher.criteria_consequences(consequence_type='early_stopping_all')
grapher.criteria_consequences(consequence_type='early_stopping_summary')