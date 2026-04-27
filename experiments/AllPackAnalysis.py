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

NAME_TO_ABBR = {
    'LunarLanderContinuous': 'LLC',
    'BipedalWalker': 'BW',
    'Ant': 'A',
    'Hopper': 'H',
    'HalfCheetah': 'HC',
    'Walker2d': 'W2d',
    'Swimmer': 'S'
}

def abbreviate(env_name: str) -> str:
    return NAME_TO_ABBR.get(env_name, env_name)

class Grapher():

    def __init__(self,library,all_packs,seeds,data_path,figure_path):
        self.library=library
        self.all_packs=all_packs
        self.seeds=seeds
        self.data_path=data_path
        self.figure_path=figure_path

    def deg_prec_eff(self):

        # Funciones para plotear datos pack-region
        def plot_pack_degradation(ax,deg_list,pack_name='',title='',not_last_pack=False):

            data = np.array(deg_list)

            bins = 8
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
            ax.set_ylabel(abbreviate(pack_name), rotation=0, labelpad=20)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_xticklabels(['0', '0.25', '0.5', '0.75', '1'])

            if not_last_pack:   
                ax.set_xticks([])
            ax.set_yticks([]) 

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
            ax.set_ylabel(abbreviate(pack_name), rotation=0, labelpad=20)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_xticklabels(['0', '0.25', '0.5', '0.75', '1'])
            ax.set_yticks([])
            ax.tick_params(axis='x', labelsize=8)
            if not last_pack:
                ax.tick_params(axis='x', labelbottom=False)
            else:
                if pack_name!='' and prec_or_eff=='prec':
            
                    legend_elements = [
                        Line2D([0], [0], marker='o', color='blue', label='Last', markersize=6, linestyle=''),
                        Line2D([0], [0], marker='^', color='orange', label='Train', markersize=6, linestyle=''),
                        Line2D([0], [0], marker='s', color='green', label='Test', markersize=6, linestyle='')
                    ]
                    ax.legend(handles=legend_elements, loc='upper center',bbox_to_anchor=(1, -0.4), ncol=3, frameon=False)
            
        # Cargar datos necesarios
        df_conf=pd.read_csv(self.data_path+'/configurations.csv')
        train_conf=str(int(df_conf.loc[df_conf["pack"] == 'all', "train_opt"].iloc[0]))
        conf_str=df_conf.loc[df_conf["pack"] == 'all', "test_cost_freq_opt"].iloc[0]
        test_conf=conf_str.split('_')[0]+'_'+conf_str.split('_')[1]+'cost'
            
        # Graficas
        fig, axs = plt.subplots(len(self.all_packs),3*3+2, figsize=(5*3,0.6*len(self.all_packs)),
                                    gridspec_kw={'width_ratios': [1,1,1,0.35,1,1,1,0.35,1,1,1]})
        plt.subplots_adjust(top=0.95,bottom=0.15,left=0.05,right=0.98, hspace=0.0,wspace=0.0)

        #---- Degradacion
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

        #---- Precision
        prec_or_eff='prec'
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
            plot_pack_criteria_prec_or_eff(axs[i,4],[last1,train1,test1][::-1],pack_name=pack.replace('pack_PPO_',''),title=title[0],last_pack=last_pack)
            plot_pack_criteria_prec_or_eff(axs[i,5],[last2,train2,test2][::-1],title=title[1],last_pack=last_pack)
            plot_pack_criteria_prec_or_eff(axs[i,6],[last3,train3,test3][::-1],title=title[2],last_pack=last_pack)

        #---- Eficacia
        prec_or_eff='eff'
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
            plot_pack_criteria_prec_or_eff(axs[i,8],[last1,train1,test1][::-1],pack_name=pack.replace('pack_PPO_',''),title=title[0],last_pack=last_pack)
            plot_pack_criteria_prec_or_eff(axs[i,9],[last2,train2,test2][::-1],title=title[1],last_pack=last_pack)
            plot_pack_criteria_prec_or_eff(axs[i,10],[last3,train3,test3][::-1],title=title[2],last_pack=last_pack)

        for i in range(len(all_packs)):
                axs[i,3].axis('off')
                axs[i,7].axis('off')

        plt.savefig(self.figure_path+'/all_setups/degradation_precision_efficiency.pdf')
 
    def criteria_comparison(self,comparison_type='how_times_best'):
        '''
        `comparison_type`: puede ser 'how_times_best', 'in_which_deg_best', 'with_what_prec_diff_best'
        '''

        df_conf=pd.read_csv(self.data_path+'/configurations.csv')
        train_conf=str(int(df_conf.loc[df_conf["pack"] == 'all', "train_opt"].iloc[0]))
        conf_str=df_conf.loc[df_conf["pack"] == 'all', "test_cost_freq_opt"].iloc[0]
        test_conf=conf_str.split('_')[0]+'_'+conf_str.split('_')[1]+'cost'

        legend_map = {
                "last":  Line2D([0], [0], marker='o', color='#0000FF', label='Last', markersize=6, linestyle=''),
                "train": Line2D([0], [0], marker='^', color='#FF8800', label='Train', markersize=6, linestyle=''),
                "test":  Line2D([0], [0], marker='s', color='green',   label='Test', markersize=6, linestyle='')
            }

        legend_groups = [["last", "train"],["last", "test"],["train", "test"]]

        # Funciones para plotear datos por pack-region
        def plot_how_many_times_better(axs, data, pack_name='', title='', not_last_pack=True):

            color_marker_map = {
                'green': 's',
                'orange': '^',
                'blue': 'o'
            }

            colors = [
                ['blue', 'orange'],
                ['blue', 'green'],
                ['orange', 'green']
            ]

            # 🔥 recorrer cada región → cada ax
            for i, (ax, (pair, (color_left, color_right))) in enumerate(zip(axs, zip(data, colors))):

                left_val, right_val = pair

                # barras enfrentadas
                ax.barh(
                    y=0,
                    width=left_val,
                    left=0,
                    color=color_left,
                    height=0.8
                )

                ax.barh(
                    y=0,
                    width=right_val,
                    left=1 - right_val,
                    color=color_right,
                    height=0.8
                )

                # marcadores extremos
                marker_left = color_marker_map.get(color_left, 'o')
                marker_right = color_marker_map.get(color_right, 'o')

                if left_val > 0:
                    ax.plot(0, 0, marker=marker_left, color='black', markersize=3)

                if right_val > 0:
                    ax.plot(1, 0, marker=marker_right, color='black', markersize=3)

                # estética por ax
                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(-0.5, 0.5)
                ax.set_yticks([])
                ax.set_ylabel(abbreviate(pack_name), rotation=0, labelpad=20)
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
                ax.set_xticklabels(['0', '0.25', '0.5', '0.75', '1'])
                ax.set_title(title)

                if not_last_pack:
                    ax.set_xticklabels([])

            # leyenda solo en último
            if not not_last_pack and pack_name != '':
                for i in range(3):
                        handles = [legend_map[k] for k in legend_groups[i]]
                        axs[i].legend(handles=handles,loc='upper center',bbox_to_anchor=(0.5, -0.4),ncol=len(handles),frameon=False)
                        

            # etiqueta global (opcional)
            axs[0].set_ylabel(abbreviate(pack_name),
                            rotation=0, labelpad=20, fontsize=8)

        def plot_in_which_deg_best(axs, data, pack_name='', title='', not_last_pack=True):

            colors = [
                ['blue', 'orange'],
                ['blue', 'green'],
                ['orange', 'green']
            ]

            color_marker_map = {
                'blue': 'o',
                'orange': '^',
                'green': 's'
            }

            bins = 10
            y_spacing = 0.5
            hist_height = 0.4

            # recorrer 3 regiones → 3 axes
            for i, (ax, pair, pair_colors) in enumerate(zip(axs, data, colors)):

                bar_height = hist_height / len(pair_colors)

                for j, (sublist, color) in enumerate(zip(pair, pair_colors)):

                    if len(sublist) == 0:
                        continue

                    sublist = np.asarray(sublist)

                    # histograma
                    hist, bin_edges = np.histogram(
                        sublist, bins=bins, range=(0, 1), density=False
                    )

                    # normalización
                    percentages = hist / hist.sum() if hist.sum() > 0 else hist

                    bin_width = bin_edges[1] - bin_edges[0]

                    # color degradado
                    base_rgb = np.array(mcolors.to_rgb(color))
                    colors_scaled = [
                        tuple(base_rgb * p + (1 - p)) for p in percentages
                    ]

                    # posición vertical dentro del ax
                    bottom = j * bar_height

                    for left, col in zip(bin_edges[:-1], colors_scaled):
                        ax.bar(
                            left,
                            bar_height,
                            width=bin_width,
                            bottom=bottom,
                            align='edge',
                            color=col,
                            edgecolor=None
                        )

                    # mediana
                    median_val = np.median(sublist)
                    mid_y = bottom + bar_height / 2
                    marker = color_marker_map.get(color, 'o')

                    ax.plot(median_val, mid_y, marker=marker,
                            color='black', markersize=3)

                # estética por ax
                ax.set_xlim(-0.05, 1.05)
                ax.set_yticks([])
                ax.set_title(title)
                ax.set_ylabel(abbreviate(pack_name), rotation=0, labelpad=20)
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
                ax.set_xticklabels(['0', '0.25', '0.5', '0.75', '1'])

                if not_last_pack:
                    ax.set_xticklabels([])

            # leyenda solo en el último si aplica
            if not_last_pack:
                for ax in axs:
                    ax.set_xticklabels([])
            else:
                if pack_name != '':
                    for i in range(3):
                        handles = [legend_map[k] for k in legend_groups[i]]
                        axs[i].legend(handles=handles,loc='upper center',bbox_to_anchor=(0.5, -0.4),ncol=len(handles),frameon=False)
                        
        def plot_with_what_prec_diff_best(axs, data, pack_name='', title='', not_last_pack=True):

            colors = [
                [["#0000FF", "#FADDBB"], ["#FF8800", "#BABAFF",]],
                [["#0000FF", "#C7EBC7"], ["green", "#BABAFF"]],
                [["#FF8800", "#C7EBC7"], ["green", "#FADDBB"]]
            ]

            color_marker_map = {
                "#0000FF": 'o',
                "#FF8800": '^',
                "green": 's',
                "#BABAFF": 'o',
                "#FADDBB": '^',
                "#C7EBC7": 's'
            }

            level_spacing = 0.3      # separacion entre subniveles
            inner_offset = 0.06       # separacion pequeña entre los dos segmentos
            cap_height = 0.04         # tamaño de los topes verticales

            axes_out = []

            # Cada region en un ax distinto
            for i, (ax, (region, region_colors)) in enumerate(zip(axs, zip(data, colors))):

                base_y = 0  # cada ax es independiente

                for j, (sublist, sub_colors) in enumerate(zip(region, region_colors)):
                    base_level_y = base_y + j * level_spacing

                    for k, (subsubdata, color) in enumerate(zip(sublist, sub_colors)):
                        offset = inner_offset if k == 0 else -inner_offset
                        y_pos = base_level_y + offset

                        if len(subsubdata) > 0:
                            q25 = np.percentile(subsubdata, 25)
                            q75 = np.percentile(subsubdata, 75)

                            ax.hlines(y_pos, q25, q75, color=color, linewidth=0.25)
                            ax.vlines(q25, y_pos - cap_height, y_pos + cap_height, color=color)
                            ax.vlines(q75, y_pos - cap_height, y_pos + cap_height, color=color)
                        else:
                            subsubdata = [2]

                        marker = color_marker_map.get(color, 'o')
                        ax.plot(np.median(subsubdata), y_pos,
                                marker=marker, color=color, markersize=3)

                ax.grid(axis='x', linestyle='--', color='gray', linewidth=0.6, alpha=0.5)
                ax.set_xlim(-0.05, 1.05)
                ax.set_yticks([])
                ax.set_ylabel(abbreviate(pack_name), rotation=0, labelpad=20)
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
                ax.set_xticklabels(['0', '0.25', '0.5', '0.75', '1'])
                ax.set_title(title)
                axes_out.append(ax)

            if not_last_pack:
                for ax in axs:
                    ax.set_xticklabels([])
            else:
                if pack_name != '':
                    for i in range(3):
                        handles = [legend_map[k] for k in legend_groups[i]]
                        axs[i].legend(handles=handles,loc='upper center',bbox_to_anchor=(0.5, -0.4),ncol=len(handles),frameon=False)

            return axes_out

        # Graficas
        fig, axs = plt.subplots(len(self.all_packs),3*3+2, figsize=(5*3,0.6*len(self.all_packs)),
                                    gridspec_kw={'width_ratios': [1,1,1,0.35,1,1,1,0.35,1,1,1]})
        plt.subplots_adjust(top=0.95,bottom=0.15,left=0.05,right=0.98, hspace=0.0,wspace=0.0)

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
                plot_how_many_times_better([axs[i,0],axs[i,4],axs[i,8]],matrix1,pack_name=pack.replace('pack_PPO_',''),title=title[0],not_last_pack=not_last_pack)
                plot_how_many_times_better([axs[i,1],axs[i,5],axs[i,9]],matrix2,title=title[1],not_last_pack=not_last_pack)
                plot_how_many_times_better([axs[i,2],axs[i,6],axs[i,10]],matrix3,title=title[2],not_last_pack=not_last_pack)

            if comparison_type=='in_which_deg_best':
                matrix1,matrix2,matrix3=DataConverter.from_df_data_to_graph_data(
                                [pack_path+'learning_regions.csv',pack_path+'df_last_truth.csv',
                                pack_path+'df_train_truth.csv',pack_path+'df_test_truth.csv',pack_path+'deg_evolution.csv'],
                                pack,which_graph=comparison_type,
                                train_conf=train_conf,test_conf=test_conf,
                                global_deg_metric=global_deg_metric,local_deg_metric=local_deg_metric
                                )
                
                plot_in_which_deg_best([axs[i,0],axs[i,4],axs[i,8]],matrix1,pack_name=pack.replace('pack_PPO_',''),title=title[0],not_last_pack=not_last_pack)
                plot_in_which_deg_best([axs[i,1],axs[i,5],axs[i,9]],matrix2,title=title[1],not_last_pack=not_last_pack)
                plot_in_which_deg_best([axs[i,2],axs[i,6],axs[i,10]],matrix3,title=title[2],not_last_pack=not_last_pack)
                
            if comparison_type=='with_what_prec_diff_best':

                matrix1,matrix2,matrix3=DataConverter.from_df_data_to_graph_data(
                                    [pack_path+'learning_regions.csv',pack_path+'df_last_truth.csv',
                                    pack_path+'df_train_truth.csv',pack_path+'df_test_truth.csv',
                                    pack_path+'df_last_prec.csv',pack_path+'df_train_prec.csv',pack_path+'df_test_prec.csv'],pack,
                                    which_graph=comparison_type,train_conf=train_conf,test_conf=test_conf,prec_metric=prec_metric
                                    )

                plot_with_what_prec_diff_best([axs[i,0],axs[i,4],axs[i,8]],matrix1,pack_name=pack.replace('pack_PPO_',''),title=title[0],not_last_pack=not_last_pack)
                plot_with_what_prec_diff_best([axs[i,1],axs[i,5],axs[i,9]],matrix2,title=title[1],not_last_pack=not_last_pack)
                plot_with_what_prec_diff_best([axs[i,2],axs[i,6],axs[i,10]],matrix3,title=title[2],not_last_pack=not_last_pack)
            
            for i in range(len(all_packs)):
                axs[i,3].axis('off')
                axs[i,7].axis('off')

        plt.savefig(self.figure_path+'/all_setups/criteria_comparison_'+comparison_type+'.pdf')
        
    def criteria_consequences(self,consequence_type='learning_curve'):

        df_conf=pd.read_csv(self.data_path+'/configurations.csv')
        conf_str=df_conf.loc[df_conf["pack"] == 'all', "test_cost_freq_opt"].iloc[0]
        test_cost_freq=conf_str.split('_')[0]+'_'+conf_str.split('_')[1]+'cost'

        legend_elements = [Line2D([0], [0], color='black', label='Truth best', linestyle='-', linewidth=1.5),
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
                for i in range(4):
                    axs[i,pack_indx].set_ylabel(r"$f(\widetilde{\pi}_t)$",fontsize=12)
                if pack_name!='':
                    axs[3,pack_indx].legend(handles=legend_elements,loc='upper center',bbox_to_anchor=(1.5, -0.3),ncol=5,frameon=False)
            for i in range(4):
                axs[i, pack_indx].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
                if i!=0:
                    axs[i, pack_indx].yaxis.get_offset_text().set_visible(False)
            axs[0,pack_indx].set_title(abbreviate(pack_name),fontsize=12)
            axs[3,pack_indx].set_xlabel('$t$',fontsize=12)
            
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
            def plot_cumulative_learning_curve_paired_diff(ax,data,color):
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
                    axs[3,pack_indx].legend(handles=legend_elements,loc='upper center',bbox_to_anchor=(1.5, -0.3),ncol=5,frameon=False)
                for i in range(4):
                    axs[i,pack_indx].set_ylabel(r"$\sum_{\rho}f(\pi^*_t)-f(\widetilde{\pi}_t)$",fontsize=10)
            for i in range(4):
                axs[i, pack_indx].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
                if i!=0:
                    axs[i, pack_indx].yaxis.get_offset_text().set_visible(False)

            axs[0,pack_indx].set_title(abbreviate(pack_name),fontsize=12)
            axs[3,pack_indx].set_xlabel(r"$t$",fontsize=12)

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
                        axs[3,pack_indx].legend(handles=legend_elements,loc='upper center',bbox_to_anchor=(1.5, -1.4),ncol=5,frameon=False)
                    for i in range(5):
                        axs[i,pack_indx].set_ylabel(r"$f(\widetilde{\pi}_t)$",fontsize=12)
                for i in range(5):
                    axs[i, pack_indx].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
                    if i!=0:
                        axs[i, pack_indx].yaxis.get_offset_text().set_visible(False)


                axs[0,pack_indx].set_title(abbreviate(pack_name),fontsize=12)
                axs[4,pack_indx].set_xlabel(r"$t$",fontsize=12)

            def plot_resource_allocation(ax, listas, colors):

                # 1. rango global
                all_data = [x for lista in listas for sub in lista for x in sub]
                min_val = min(all_data)
                max_val = max(all_data)

                # bins no uniformes
                bins = np.array([min_val,min_val + 0.25 * (max_val - min_val),min_val + 0.75 * (max_val - min_val),max_val])

                n_groups = len(listas)
                bar_height = (max_val - min_val) * 0.05
                spacing = bar_height * 1.2
                outer_margin = (max_val - min_val) * 0.08

                # 2. histogramas
                histogramas = []
                for lista_de_listas in listas:
                    datos = [x for sub in lista_de_listas for x in sub]
                    counts, _ = np.histogram(datos, bins=bins)
                    histogramas.append(counts)

                # 3. dibujar
                for b in range(len(bins) - 1):

                    y_center = (bins[b] + bins[b+1]) / 2

                    total_height = (n_groups - 1) * spacing
                    start = y_center - total_height / 2

                    for i, counts in enumerate(histogramas):
                        y = start + i * spacing
                        ax.barh(y,counts[b],height=bar_height,left=0,color=colors[i],align='center')

                # 4. estetica
                ax.set_ylim(min_val - outer_margin, max_val + outer_margin)  
                ax.set_xlim(0, max(max(h) for h in histogramas) * 1.1)

                bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
                ax.set_yticks(bin_centers)

                ax.set_xlabel('Number of times')

                if pack_indx == 0:
                    ax.set_ylabel('Truth values')
                    ax.set_yticklabels(["low", "middle", "high"])
                    ax.legend(handles=legend_elements,loc='upper center',bbox_to_anchor=(2, -0.3),ncol=5,frameon=False)
                else:
                    ax.set_yticklabels(["", "", ""])    

            if early_stopping_graph=='summary':
                plot_resource_allocation(axs[pack_indx],
                                            [matrix_best,matrix_last,matrix_train_default,matrix_test_default,matrix_test],
                                            ['black','blue','orange','purple','green'])
                axs[pack_indx].set_title(abbreviate(pack_name))


        if consequence_type in ['learning_curve','cummulative_diff']:
            fig, axs = plt.subplots(4,len(self.all_packs), figsize=(3*len(self.all_packs),6),sharex='col',sharey='col')
            plt.subplots_adjust(top=0.95,bottom=0.1,left=0.05,right=0.98, hspace=0.05,wspace=0.15)
        if consequence_type=='early_stopping_all':
            fig, axs = plt.subplots(5,len(self.all_packs), figsize=(2.5*len(self.all_packs),9),sharex='col',sharey='col')
            plt.subplots_adjust(top=0.95,bottom=0.1,left=0.05,right=0.98, hspace=0.05,wspace=0.25)
        if consequence_type=='early_stopping_summary':
            fig, axs = plt.subplots(1,len(self.all_packs), figsize=(2*len(self.all_packs),3))
            plt.subplots_adjust(top=0.9,bottom=0.3,left=0.07,right=0.98, hspace=0.0,wspace=0.05)

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
grapher.deg_prec_eff()
grapher.criteria_comparison()
grapher.criteria_comparison(comparison_type='in_which_deg_best')
grapher.criteria_comparison(comparison_type='with_what_prec_diff_best')
grapher.criteria_consequences()
grapher.criteria_consequences(consequence_type='cummulative_diff')
grapher.criteria_consequences(consequence_type='early_stopping_all')
grapher.criteria_consequences(consequence_type='early_stopping_summary')