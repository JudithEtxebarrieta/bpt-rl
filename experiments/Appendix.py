from Main import *
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize


data_path='experiments/results/data'
figure_path='experiments/results/figures'
library='SB3'

seeds=list(range(1,31))
all_packs=[ # ClassicControl
            #'pack_PPO_Pendulum',
            # Box2D
            'pack_PPO_LunarLanderContinuous',
            'pack_PPO_BipedalWalker',
            # MuJoCo
            'pack_PPO_Swimmer',
            'pack_PPO_Hopper',
            'pack_PPO_HalfCheetah',
            'pack_PPO_Walker2d' , 
            'pack_PPO_Ant'         
                            ]

packs_with_norm=[ 
            # Box2D
            'pack_PPO_BipedalWalker'
            # MuJoCo
            # 'pack_PPO_HalfCheektah',
            # 'pack_PPO_Hopper',
            # 'pack_PPO_Ant',
            # 'pack_PPO_Walker2d'            
                            ]*2


global_deg_metric='norm_from_mean_worsening_to_improvement'
local_deg_metric='reward_diff'
prec_metric='relative_perc_criteria_best'
eff_metric='first_time_to_same_reward'

NAME_TO_ABBR = {
    'Pendulum': 'P',
    'LunarLanderContinuous': 'LLC',
    'BipedalWalker': 'BW',
    'Ant': 'A',
    'Hopper': 'H',
    'HalfCheetah': 'HC',
    'Walker2d': 'W2d',
    'Swimmer': 'S',
    'BipedalWalkerNorm': 'BW',
    'AntNorm': 'A',
    'HopperNorm': 'H',
    'HalfCheetahNorm': 'HC',
    'Walker2dNorm': 'W2d'
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

    def deg_with_without_norm(self,current_packs):

        #TODO: debo ejecutarlo cuando ya tengo todos los datos con normalize=true

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

                # Graficas
    

        # Grafica de degradaciones
        fig, axs = plt.subplots(len(current_packs),3*2+1, figsize=(5*2,0.6*len(current_packs)),
                                    gridspec_kw={'width_ratios': [1,1,1,0.35,1,1,1]})
        plt.subplots_adjust(top=0.95,bottom=0.15,left=0.05,right=0.98, hspace=0.0,wspace=0.0)

        for i,pack in enumerate(current_packs):

            pack_path=self.data_path+'/'+pack.replace('pack',self.library)+'/'
            pack_norm_path=self.data_path+'/'+pack.replace('pack',self.library)+'Norm'+'/'

            # Obtener los datos de degradacion por region de este pack
            deg1,deg2,deg3=DataConverter.from_df_data_to_graph_data(
                                    [pack_path+'learning_regions.csv',pack_path+'deg_evolution.csv'],pack,'deg_distribution',
                                    global_deg_metric=global_deg_metric,local_deg_metric=local_deg_metric
                                ) 
            deg1_norm,deg2_norm,deg3_norm=DataConverter.from_df_data_to_graph_data(
                                    [pack_norm_path+'learning_regions.csv',pack_norm_path+'deg_evolution.csv'],pack,'deg_distribution',
                                    global_deg_metric=global_deg_metric,local_deg_metric=local_deg_metric
                                ) 

            # Dibujar distribucion de degradacion de las tres regiones
            title=[['Initialization','Learning','Convergence']if i==0 else ['']*3][0]
            not_last_pack=[False if i==len(current_packs)-1 else True][0]
            plot_pack_degradation(axs[i,0],deg1,pack_name=pack.replace('pack_PPO_',''),title=title[0],not_last_pack=not_last_pack)
            plot_pack_degradation(axs[i,1],deg2,title=title[1],not_last_pack=not_last_pack)
            plot_pack_degradation(axs[i,2],deg3,title=title[2],not_last_pack=not_last_pack)
            plot_pack_degradation(axs[i,4],deg1_norm,pack_name=pack.replace('pack_PPO_',''),title=title[0],not_last_pack=not_last_pack)
            plot_pack_degradation(axs[i,5],deg2_norm,title=title[1],not_last_pack=not_last_pack)
            plot_pack_degradation(axs[i,6],deg3_norm,title=title[2],not_last_pack=not_last_pack)

            axs[i,3].axis('off')

        plt.savefig(self.figure_path+'/all_setups/appendix_norm_influence_deg.pdf')

    def estimations(self,train_white=False):


        def plot_analysis_per_seed(ax,truth,cols_test,cols_train,first_row,last_row,first_column):

            # Ground truth
            ax.plot(range(len(truth)),np.array(truth)-np.array(truth), color='black', label='truth')

            # Estimates with validation
            greens = plt.cm.Greens(np.linspace(0.3, 0.9, len(cols_test.columns)))
            oranges = plt.cm.Oranges(np.linspace(0.3, 0.9, len(cols_train.columns)))

            for col, color in zip(cols_test.columns[::-1], greens):
                ax.plot(range(len(truth)), abs(np.array(cols_test[col].values)-np.array(truth)), color=color, label=col.split('_')[0]+' test',linewidth=1)
            for col, color in zip(cols_train.columns[::-1], oranges):
                if train_white:
                    ax.plot(range(len(truth)), abs(np.array(cols_train[col].values)-np.array(truth)), color='white',alpha=0,linewidth=1)
                else:
                    ax.plot(range(len(truth)), abs(np.array(cols_train[col].values)-np.array(truth)), color=color, label=col.split('_')[0]+' train',linewidth=1)

            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            offset = ax.yaxis.get_offset_text()
            offset.set_x(-0.15)

            if first_row:
                ax.set_title(abbreviate(pack.split('_')[2]))
            else:
                ax.yaxis.get_offset_text().set_visible(False)
            if last_row:
                ax.set_xlabel(r'$t$')
                ax.tick_params(labelbottom=True)
            else:
                ax.tick_params(labelbottom=False)
            if first_column:
                ax.set_ylabel(r'$|\widetilde{f}(\pi_t)-f(\pi_t)|$')

            if last_row and first_column:
                leg=ax.legend(
                        loc='upper center',
                        bbox_to_anchor=(4, -0.5),
                        ncol=len(ax.get_legend_handles_labels()[0]),
                        frameon=False
                    )
                for line in leg.get_lines():
                    line.set_linewidth(3)

        n_seeds=5

        fig, axs = plt.subplots(n_seeds,len(self.all_packs), figsize=(2.5*len(self.all_packs),1.2*n_seeds),sharex='col',sharey='col')
        plt.subplots_adjust(top=0.95,bottom=0.15,left=0.05,right=0.98, hspace=0.1,wspace=0.2)


        for i,pack in tqdm(enumerate(self.all_packs)):
            df_truth=pd.read_csv(self.data_path+'/'+pack.replace('pack','SB3')+'/df_last_truth.csv')
            df_test=pd.read_csv(self.data_path+'/'+pack.replace('pack','SB3')+'/df_test_all_seq_est.csv')
            df_train=pd.read_csv(self.data_path+'/'+pack.replace('pack','SB3')+'/df_train_all_seq_est.csv')

            for seed in range(1,n_seeds+1):

                first_row=[True if seed==1 else False][0]
                last_row=[True if seed==n_seeds else False][0]
                first_column=[True if i==0 else False][0]

                truth= df_truth[pack+str(seed)].tolist()
                cols_test = df_test[[c for c in df_test.columns if c.endswith('_seed'+str(seed))]]
                cols_test=cols_test.applymap(lambda x: DataConverter.compress_decompress_list(x,compress=False)[0])
                cols_train = df_train[[c for c in df_train.columns if c.endswith('_seed'+str(seed))]]

                plot_analysis_per_seed(axs[seed-1,i],truth,cols_test,cols_train,first_row,last_row,first_column)

        if train_white:
            plt.savefig(self.figure_path+'/all_setups/appendix_estimations_only_test.pdf')
        else:
            plt.savefig(self.figure_path+'/all_setups/appendix_estimations.pdf')

    def regions_with_deg(self):

        def plot_analysis_per_seed(ax1,ax2,pack,seed,
                                    global_deg_metric=global_deg_metric,local_deg_metric=local_deg_metric,
                                    first_row=False,first_column=False,last_row=False):

            colors=list(mcolors.TABLEAU_COLORS.keys())

            # Evolucion de degradacion
            deg=pd.read_csv(self.data_path+'/'+pack.replace('pack','SB3')+'/deg_evolution.csv')[pack+str(seed)+'_'+global_deg_metric+'_'+local_deg_metric].tolist()
            
            ax1.imshow(np.array(deg)[np.newaxis, :], cmap='gray_r', vmin=0, vmax=1,aspect='auto', interpolation='nearest')
            ax1.set_xticks([])
            ax1.set_yticks([])
            
            # Evolucion de estimaciones truth
            truth_last=pd.read_csv(self.data_path+'/'+pack.replace('pack','SB3')+'/df_last_truth.csv')[pack+str(seed)].tolist()
            ax2.plot(range(len(truth_last)), truth_last, label="Truth",color=colors[0],linewidth=1)

            # Limites a (donde empieza a aprender) y b (donde se termina de aprender)
            df_limits=pd.read_csv(self.data_path+'/'+pack.replace('pack','SB3')+'/learning_regions.csv')
            a,b=df_limits.loc[(df_limits['pack_seed'] == pack + str(seed)),['a', 'b']].iloc[0]

            ax2.axvline(x=a, color='red', linestyle='-', linewidth=2,label='start learning')
            ax2.axvline(x=b, color='black', linestyle='-', linewidth=2,label='end learning')
            ax2.axvline(x=int(len(truth_last)/1.5), color='black', linestyle='--', linewidth=2,label='default end')
            ax2.set_xlim(0,len(truth_last))
            ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            offset = ax2.yaxis.get_offset_text()
            offset.set_x(-0.15)

            if first_row:
                ax1.set_title(abbreviate(pack.split('_')[2]))
            if last_row:
                ax2.set_xlabel(r'$t$')
                ax2.tick_params(labelbottom=True)
            else:
                ax2.tick_params(labelbottom=False)
            if first_column:
                ax2.set_ylabel(r'$f(\pi_t)$')

        n_seeds=5

        fig, axs = plt.subplots(n_seeds*2,len(self.all_packs), figsize=(2.5*len(self.all_packs),1.2*n_seeds),
                                height_ratios=[0.2,1]*n_seeds)
        plt.subplots_adjust(top=0.95,bottom=0.15,left=0.05,right=0.98, hspace=0.2,wspace=0.2)

        # compartir ejes solo para filas impares
        for col in range(len(self.all_packs)):
            
            # referencia = primera fila impar
            ref_ax = axs[1, col]
            
            for row in range(3, n_seeds*2, 2):
                axs[row, col].sharex(ref_ax)
                axs[row, col].sharey(ref_ax)

        for i,pack in enumerate(self.all_packs):
            for seed in range(1,n_seeds+1):#TODO: puede que para meter en el paper tenga que reducir el numero de semillas, igual a 5
                first_row=[True if seed==1 else False][0]
                last_row=[True if seed==n_seeds else False][0]
                first_column=[True if i==0 else False][0]
                j=seed-1
                plot_analysis_per_seed(axs[j*2-n_seeds*2*((j+1)//n_seeds),i],axs[j*2+1-n_seeds*2*((j+1)//n_seeds),i],pack,seed,
                                       first_row=first_row,last_row=last_row,first_column=first_column)
                
        legend_elements = [
                    Patch(facecolor='black', edgecolor='black', label=r'$\delta_t=1$'),
                    Patch(facecolor='white', edgecolor='black', label=r'$\delta_t=0$'),
                    Line2D([0], [0], color='red', lw=2, label=r'$a$'),
                    Line2D([0], [0], color='black', lw=2, label=r'$b$'),
                    Line2D([0], [0], color='black', lw=2, linestyle='--', label=r'$t_{max}$')
                    
                ]
        fig.legend(handles=legend_elements,loc='lower center',ncol=5,frameon=False,bbox_to_anchor=(0.18, 0.005))

        plt.savefig(self.figure_path+'/all_setups/appendix_regions.pdf')    

    def criteria_configurations(self):


        def graph_deg_criteria_conf_by_regions(self,ax1_train,ax2_train,ax3_train,ax1_test,ax2_test,ax3_test,
                                               pack,first_column,prec_metric='relative_perc_criteria_best'):

            # Leer configuraciones generales optimas para poder las marcar en las graficas
            df_conf=pd.read_csv(self.data_path+'/configurations.csv')
            general_train_n_ep=df_conf.loc[df_conf["pack"] == 'all', "train_opt"].iloc[0]
            conf_str=df_conf.loc[df_conf["pack"] == 'all', "test_cost_freq_opt"].iloc[0]
            general_test_n_ep=int(conf_str.split('_')[0])
            general_test_freq=float(conf_str.split('_')[1])
            default_train_n_ep=int(df_conf.loc[df_conf["pack"] == pack, "train_default"].iloc[0])

            #----- Cuadricula de grafica interna

            prec1_simple,prec2_simple,prec3_simple=[],[],[]

            # 1) Train conf_prec
            def train_conf_precCI(ax,listas,n_ep_list,nombres=None,optimal_conf=None,title=''):

                for y, data in zip(range(len(listas)), listas):
                    if optimal_conf!=None:
                        if int(optimal_conf)==int(n_ep_list[::-1][len(listas)-1-y]) :
                            ax.axhline(y, color='yellow', linewidth=10,alpha=0.3)
                    if default_train_n_ep==int(n_ep_list[y]):
                        ax.axhline(y, color='black', linewidth=10,alpha=0.05)
                    if general_train_n_ep==int(n_ep_list[::-1][len(listas)-1-y]) :
                        ax.axhline(y, color="red", linewidth=10,alpha=0.3)

                    ax.hlines(y, np.percentile(data, 25), np.percentile(data, 75), color='black')
                    ax.vlines(np.percentile(data, 25), y - 0.2, y + 0.2, color='black')
                    ax.vlines(np.percentile(data, 75), y - 0.2, y + 0.2, color='black')
                    ax.plot(np.median(data), y, 'o', color='black')

                ax.set_xlim(-0.1,1.1)
                ax.grid(axis='x', linestyle='--',alpha=0.4)
                if nombres!=None:
                    ax.set_yticks(range(len(n_ep_list)), n_ep_list)
                    if first_column:
                        ax.set_ylabel(r'train $\kappa$')

                else:
                    ax.set_yticks([])

                ax.set_title(title)
                ax.set_xticklabels([])
        
            def obtain_best_train_conf(prec1,prec2,prec3,n_ep_list):

                q5_learning=[np.percentile(i, 25)+np.percentile(j, 25)+np.percentile(k, 25) for i,j,k in zip(prec1,prec2,prec3)]
                return n_ep_list[q5_learning.index(max(q5_learning))]
            
            prec1,prec2,prec3,conf_list=DataConverter.from_df_data_to_graph_data(
                [self.data_path+'/'+pack.replace('pack','SB3')+'/learning_regions.csv',self.data_path+'/'+pack.replace('pack','SB3')+'/df_train_prec.csv'],pack,which_graph='train_conf_prec',
                prec_metric=prec_metric
            )
    
            n_ep_train=obtain_best_train_conf(prec1,prec2,prec3,conf_list)
            train_conf_precCI(ax1_train,prec1,conf_list,nombres=True,optimal_conf=n_ep_train)
            train_conf_precCI(ax2_train,prec2,conf_list,optimal_conf=n_ep_train,title=abbreviate(pack.split('_')[2]))
            train_conf_precCI(ax3_train,prec3,conf_list,optimal_conf=n_ep_train)

            conf_list=[int(n_ep) for n_ep in conf_list]
            prec1_simple.append(prec1[conf_list.index(int(general_train_n_ep))])
            prec2_simple.append(prec2[conf_list.index(int(general_train_n_ep))])
            prec3_simple.append(prec3[conf_list.index(int(general_train_n_ep))])

            # 2) Test conf_prec_cost
            def test_conf_precCI_costColor(ax,prec_matrix,cost_matrix,n_ep_list,freq_list,nombres=None,optimal_conf=[None,None]):

                # Grafica
                current_height = 0
                segment_labels = []
                region_centers = []

                for i in range(len(n_ep_list)):
                    
                    region_start = current_height
                    for j in range(len(freq_list)):

                        if int(optimal_conf[0])==int(n_ep_list[i]) and float(optimal_conf[1])==float(freq_list[j]):
                            ax.axhline(current_height, color='yellow', linewidth=10,alpha=0.3)

                        if general_test_n_ep==float(n_ep_list[i]) and general_test_freq==float(freq_list[j]):
                            ax.axhline(current_height, color="red", linewidth=10,alpha=0.3)

                        datos = prec_matrix[i][j]
                           
                        ax.hlines(current_height, np.percentile(datos, 25), np.percentile(datos, 75), color='black')
                        ax.vlines([np.percentile(datos, 25), np.percentile(datos, 75)], current_height-0.2, current_height+0.2, color='black')
                        ax.scatter(np.median(datos), current_height, color='black', marker='o', zorder=3)
                    
                        segment_labels.append(freq_list[j])
                        current_height += 1
                    
                    region_end = current_height - 1
                    region_centers.append((region_start + region_end)/2)
                    
                    if i < len(n_ep_list)-1: # Linea separadora de regiones
                        ax.axhline(current_height-0.5, color='black', linewidth=1)
                ax.grid(axis='x', linestyle='--', alpha=0.5)
                ax.set_yticks(range(len(segment_labels)), segment_labels)

                if nombres!=None:
                    for center, name in zip(region_centers, n_ep_list):
                        ax.text(-0.25, center, name,transform=ax.get_yaxis_transform(),ha='right', va='center')
                    if first_column:
                        ax.set_ylabel(r'test $(\kappa,\varphi_c)$',labelpad=35)
                else:
                    ax.set_yticks([])
      

                ax.set_xlim(-0.1,1.1)
                ax.set_title("")
                ax.invert_yaxis()

            def obtain_best_test_conf(prec1,prec2,prec3,cost1,cost2,cost3,n_ep_list,freq_list,threshold=0.2):

                q5_learning=[]
                for idx_cost in range(len(prec1)):
                    q5_sublist=[]
                    for i,j,k in zip(prec1[idx_cost],prec2[idx_cost],prec3[idx_cost]):
                        q5_sublist.append(np.percentile(i, 25)+np.percentile(j, 25)+np.percentile(k, 25))
                    q5_learning.append(q5_sublist)

                max_cost_learning=[]
                for idx_cost in range(len(cost1)):
                    max_sublist=[]
                    for i,j,k in zip(cost1[idx_cost],cost2[idx_cost],cost3[idx_cost]):
                        max_sublist.append(max(i+j+k))
                    max_cost_learning.append(max_sublist)

               
                i, j = max(((i, j) for i, sub in enumerate(q5_learning) for j, v in enumerate(sub)),key=lambda x: q5_learning[x[0]][x[1]])
                return int(n_ep_list[i]), freq_list[j]

            prec1,prec2,prec3,n_ep_list,freq_list=DataConverter.from_df_data_to_graph_data(
                [self.data_path+'/'+pack.replace('pack','SB3')+'/learning_regions.csv',self.data_path+'/'+pack.replace('pack','SB3')+'/df_test_prec.csv'],pack,which_graph='test_conf_prec_cost',
                prec_metric=prec_metric,n_ep_type='with_cost_freq'
            )
            cost1,cost2,cost3,n_ep_list,freq_list=DataConverter.from_df_data_to_graph_data(
                [self.data_path+'/'+pack.replace('pack','SB3')+'/learning_regions.csv',self.data_path+'/'+pack.replace('pack','SB3')+'/df_test_cost.csv'],pack,which_graph='test_conf_prec_cost',
                prec_metric=prec_metric,n_ep_type='with_cost_freq'
            )

            n_ep_test,freq_test=obtain_best_test_conf(prec1,prec2,prec3,cost1,cost2,cost3,n_ep_list,freq_list)
            test_conf_precCI_costColor(ax1_test,prec1,cost1,n_ep_list,freq_list,nombres=True,optimal_conf=[n_ep_test,freq_test])
            test_conf_precCI_costColor(ax2_test,prec2,cost2,n_ep_list,freq_list,optimal_conf=[n_ep_test,freq_test])
            test_conf_precCI_costColor(ax3_test,prec3,cost3,n_ep_list,freq_list,optimal_conf=[n_ep_test,freq_test])


        fig, axs = plt.subplots(8,11, figsize=(1.8*11,2*8),
                                gridspec_kw={'width_ratios': [1]*3+[0.5]+[1]*3+[0.5]+[1]*3,
                                             'height_ratios': [0.1,0.5]+[0.05]+[0.1,0.5]+[0.05]+[0.1,0.5]})
        plt.subplots_adjust(top=0.95,bottom=0.05,left=0.05,right=0.98, hspace=0.1,wspace=0)


        for i,pack in tqdm(enumerate(self.all_packs)):
            first_column=[True if i%3==0 else False][0]
            if i!=len(self.all_packs)-1:
                graph_deg_criteria_conf_by_regions(self,
                                                axs[3*(i//3),4*(i%3)],axs[3*(i//3),1+4*(i%3)],axs[3*(i//3),2+4*(i%3)],
                                                axs[1+3*(i//3),4*(i%3)],axs[1+3*(i//3),1+4*(i%3)],axs[1+3*(i//3),2+4*(i%3)],pack,first_column)
            else:
                graph_deg_criteria_conf_by_regions(self,
                                                axs[6,4],axs[6,5],axs[6,6],
                                                axs[7,4],axs[7,5],axs[7,6],pack,first_column)
            

        for i in range(8):
            axs[i,3].axis('off')
            axs[i,7].axis('off')
        for i in range(11):
            axs[2,i].axis('off')
            axs[5,i].axis('off')

        for i in [0,1,2,8,9,10]:
            axs[6,i].axis('off')
            axs[7,i].axis('off')

        legend_elements = [Line2D([0], [0], color='yellow', lw=6, alpha=0.3, label='max p25_initialization+p25_learning+p25_stabilization'),
                                Line2D([0], [0], color="red", lw=6, alpha=0.3, label='in general (all setups)'), 
                                Line2D([0], [0], color='grey', lw=6, alpha=0.3, label='by default')
                                ]
        axs[6,8].legend(
                    handles=legend_elements,
                    title='',
                    loc='upper center',
                    bbox_to_anchor=(1.5, 0),
                    ncol=1,         
                    frameon=False,
                    fontsize=12
                )





        plt.savefig(self.figure_path+'/all_setups/appendix_configurations.pdf')

    def val_cost_throughout_sequence(self):

        # Escala de colores
        cmap_gwr = LinearSegmentedColormap.from_list('green_white_red',
            [(0.0, 'darkgreen'),(0.3, 'white'),(1.0, 'darkred')])

        norm_gwr = Normalize(vmin=0, vmax=1)

        def plot_analysis_per_seed(ax1,ax2,axs_cost,list1,list2,df,last_row,pack):
            
            def normalize(x):
                x = np.array(x, dtype=float)
                if x.max() == x.min():
                    return np.array([1]*len(x))
                return (x - x.min()) / (x.max() - x.min())

            # Normalizar truth y ep_lens
            list1_norm = normalize(list1)
            list2_norm = normalize(list2)

            # Grafica
            #--- truth
            ax1.imshow(list1_norm[np.newaxis, :],cmap='gray_r',aspect='auto',interpolation='nearest',vmin=0,vmax=1)
            ax1.set_ylabel(r'$f(\pi_t)$', rotation=0,fontsize=8)
            ax1.yaxis.set_label_coords(-0.05, 0)

            #--- ep_lens
            ax2.imshow(list2_norm[np.newaxis, :],cmap='gray_r',aspect='auto',interpolation='nearest',vmin=0,vmax=1)
            ax2.set_ylabel(r'$|\tau|$', rotation=0,fontsize=8)
            ax2.yaxis.set_label_coords(-0.05, 0)

            #--- val_cost
            for i, col in enumerate(df.columns):
                values = np.array(df[col].values, dtype=float)

                # limitar valores >1 a 1 para el color
                values_clip = np.clip(values, 0, 1)
                
                axs_cost[i].imshow(values_clip[np.newaxis, :],cmap=cmap_gwr,norm=norm_gwr,aspect='auto',interpolation='nearest')
                axs_cost[i].set_ylabel(col.split('_')[0], rotation=0,fontsize=8)
                axs_cost[i].yaxis.set_label_coords(-0.05, 0)

            # Limpieza visual
            all_axs=[ax1,ax2]+list(axs_cost)
            for ax in all_axs:
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

            if last_row:
                for ax in all_axs[:-1]:
                    ax.set_xticks([])
                all_axs[-1].set_xlabel(r'$t$')
            else:
                for ax in all_axs[:-2]:
                    ax.set_xticks([])
                all_axs[-1].set_xlabel('')
            all_axs[-2].tick_params(axis='x', labelsize=8,length=0.5)
            all_axs[-1].tick_params(axis='x', labelsize=8,length=0.5)
            axs_cost[1].text(-0.09, 0.5,abbreviate(pack.replace('pack_PPO_','')),rotation=90,va='center',ha='center',transform=axs_cost[1].transAxes,fontsize=8)


  
        fig, axs = plt.subplots(8*len(self.all_packs)+len(self.all_packs)-1,1, figsize=(5,1*len(self.all_packs)),
                                gridspec_kw={'height_ratios': [1]*8+[2]+[1]*8+[2]+[1]*8+[2]+[1]*8+[2]+[1]*8+[2]+[1]*8+[2]+[1]*8})
        plt.subplots_adjust(top=0.99,bottom=0.13,left=0.1,right=0.98, hspace=0.1,wspace=0.2)

        what_seed=1

        for i,pack in tqdm(enumerate(self.all_packs)):
            df_truth=pd.read_csv(self.data_path+'/'+pack.replace('pack','SB3')+'/df_last_truth.csv')
            df_test=pd.read_csv(self.data_path+'/'+pack.replace('pack','SB3')+'/df_test_all_seq_cost_len.csv')
               
            truth=df_truth[pack+str(what_seed)].tolist()
            ep_lens=df_test['ep_len_seed'+str(what_seed)].tolist()
            val_costs=df_test[[c for c in df_test.columns if c.endswith('val_ep_seed'+str(what_seed))]]

            last_row=[True if i==len(self.all_packs)-1 else False][0]
            plot_analysis_per_seed(axs[9*i],axs[1+9*i],axs[2+9*i:9+9*i],truth[1:],ep_lens[1:],val_costs.iloc[1:],last_row,pack)

        for i in [8,17,26,35,44,53]:
            axs[i].axis('off')

        # Leyenda
        sm = plt.cm.ScalarMappable(cmap=cmap_gwr, norm=norm_gwr)
        sm.set_array([])
        cmap_bw = LinearSegmentedColormap.from_list("white_black", ["white", "black"])
        sm2 = plt.cm.ScalarMappable(cmap=cmap_bw, norm=plt.Normalize(0, 1))
        sm2.set_array([])

        # posición base (ajústalo si quieres subir/bajar todo)
        y = 0.06
        h = 0.01
        w = 0.35

        # izquierda
        cax1 = fig.add_axes([0.08+0.05, y, w, h])
        cbar1 = fig.colorbar(sm, cax=cax1, orientation='horizontal')
        cbar1.set_label('Validation-to-learning iter. cost ratio', fontsize=8)
        cbar1.ax.tick_params(labelsize=8)

        # derecha
        cax2 = fig.add_axes([0.54+0.05, y, w, h])
        cbar2 = fig.colorbar(sm2, cax=cax2, orientation='horizontal')
        cbar2.set_label(r'Normalized $f(\pi_t)$ or mean $|\tau|$', fontsize=8)
        cbar2.ax.tick_params(labelsize=8)

        plt.savefig(self.figure_path+'/all_setups/appendix_val_cost_throughout_sequence.pdf')

    def cost_of_cost_driven_test(self):

        list_n_ep=[500,250,100,50,25,5]
        list_cost=[0.05,0.1,0.15,0.2]
        colors=plt.rcParams['axes.prop_cycle'].by_key()['color']

        # GRAFICA 1: evolucion de ratio de coste

        def plot_cost_curve_with_CI(ax,df,color,cost,first_column,last_row):
            ax.plot(np.arange(len(df)), df.quantile(0.5, axis=1), color=color,linewidth=1)
            ax.fill_between(np.arange(len(df)),df.quantile(0.25, axis=1),df.quantile(0.75, axis=1),color=color,alpha=0.3)
            ax.set_ylim(0,cost)
            ax.set_yticks([0,cost])
            ax.set_xlim(0,df.shape[0])

            if not first_column:
                ax.tick_params(labelleft=False)
            if first_column:
                ax.set_ylabel(r'$c_{val}/c_{learn}$')
            if not last_row:
                ax.tick_params(labelbottom=False)
            if last_row:
                ax.set_xlabel(r'$t$')
  
        def plot_iters_upper_cost_threshold(ax,list):

            img = np.array(list, dtype=int)[np.newaxis, :]
            cmap = ListedColormap(["white", "red"])
            ax.imshow(img,aspect='auto',cmap=cmap,interpolation='nearest')

            # limpieza visual
            ax.set_yticks([])
            ax.set_xticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        fig, axs = plt.subplots(len(list_cost)*2+len(list_cost)-1,len(self.all_packs), figsize=(1.6*len(self.all_packs),1*len(list_cost)),
                                gridspec_kw={'height_ratios': [0.2, 1,0.1]*(len(list_cost)-1)+[0.2,1]})
        plt.subplots_adjust(top=0.91,bottom=0.15,left=0.07,right=0.98, hspace=0.01,wspace=0.1)

        _, ncols = axs.shape
        for row in [1, 4, 7, 10]:
            ref_ax = axs[row, 0]  # eje referencia de la fila

            for col in range(1, ncols):
                axs[row, col].sharey(ref_ax)

        for i,pack in tqdm(enumerate(self.all_packs)):
            first_column=[True if i==0 else False][0]

            df_costs=pd.read_csv(data_path+'/'+pack.replace('pack','SB3')+'/df_test_cost.csv')

            for j,cost in enumerate(list_cost):
                last_row=[True if j==len(list_cost)-1 else False][0]
                up_to_cost=[0]*df_costs.shape[0]

                # Primero dibujar curvas de evolucion de coste
                for k,n_ep in enumerate(list_n_ep):

                    df_current=df_costs[[c for c in df_costs.columns if f"_{n_ep}_{cost}cost_" in c]]

                    # Dibujar curva
                    plot_cost_curve_with_CI(axs[2*j+1+j,i],df_current,colors[k],cost,first_column,last_row)
                    # Mirar en que iteraciones de la secuencia nos hemos pasado del umbral de coste
                    mask = (df_current > cost).any(axis=1)
                    up_to_cost = np.array(up_to_cost)+ np.array(mask.astype(int).tolist())

                # Despues dibujar sobre esta la grafica auxiliar que indica dodnde nos hemos pasado
                plot_iters_upper_cost_threshold(axs[2*j+j,i],up_to_cost)

            axs[0,i].set_title(abbreviate(pack.replace('pack_PPO_','')),fontsize=8)

        for i in [2,5,8]:
            for j in range(7):
                axs[i,j].axis('off')

        # Leyenda
        legend_elements = []

        # lineas de colores
        for c, n in zip(colors[:len(list_n_ep)], [ r'$\kappa$='+str(i) for i in list_n_ep]):
            legend_elements.append(Line2D([0], [0], color=c, lw=2, label=str(n)))

        # linea roja extra
        legend_elements.append(Line2D([0], [0], color='red', lw=2, label=r'$\varphi_c$ is surpassed'))
        axs[len(list_cost)*2+len(list_cost)-2,0].legend(handles=legend_elements,
            loc='upper center',bbox_to_anchor=(2.5, -0.55),ncol=len(legend_elements), frameon=False)
                
        plt.savefig(self.figure_path+'/all_setups/appendix_cost_of_cost_driven_test.pdf')


        #GRAFICA 2: boxplot de ratio de costes por umbral-setup

        def plot_cost_boxplots_per_pack(ax,data,list_costs,pack,first_column):

            for y in list_cost:
                ax.axhline(y=y, color='gray', linestyle='-', linewidth=1)
            ax.boxplot(
                    data,
                    medianprops=dict(color='red', linewidth=2),
                    flierprops=dict(
                        marker='_',        
                        markersize=8,
                        markeredgecolor='black',
                        markeredgewidth=0.5
                    )
                )
            ax.set_xticks(list(range(1,len(list_costs)+1)))
            ax.set_xticklabels([str(cost) for cost in list_costs])
            ax.set_title(abbreviate(pack.replace('pack_PPO_','')),fontsize=8)

            if not first_column:
                ax.set_yticks([])
            if first_column:
                ax.set_ylabel(r'$c_{val}/c_{learn}$')
            ax.set_xlabel(r'$\varphi_c$')

        fig, axs = plt.subplots(1,len(self.all_packs), figsize=(1.8*len(self.all_packs),2))
        plt.subplots_adjust(top=0.85,bottom=0.21,left=0.07,right=0.98, hspace=0.1,wspace=0.1)

        for i,pack in tqdm(enumerate(self.all_packs)):
            first_column=[True if i==0 else False][0]

            df_costs=pd.read_csv(data_path+'/'+pack.replace('pack','SB3')+'/df_test_cost.csv')

            all_ratios_per_cost=[]
            for j,cost in enumerate(list_cost):

                # Guardar todos los ratios de coste para despues dibujar boxplot
                all_ratios=[]
                for k,n_ep in enumerate(list_n_ep):
                    df_current=df_costs[[c for c in df_costs.columns if f"_{n_ep}_{cost}cost_" in c]]
                    all_ratios+=df_current.values.ravel().tolist()

                # Ratios por coste
                all_ratios_per_cost.append([1 if ratio > 1 else ratio for ratio in all_ratios])

            # Dibujar grafica de boxplots por del pack actual
            plot_cost_boxplots_per_pack(axs[i],all_ratios_per_cost,list_cost,pack,first_column)

        plt.savefig(self.figure_path+'/all_setups/appendix_cost_of_cost_driven_test_boxplots.png')

        # GRAFICA 3: boxplot por pack-umbral-n_ep junto a IQR-median de prec

        def plot_cost_boxplots_per_pack_threshold(ax,data,prec_stats,list_n_ep,pack,cost,first_column,first_row,last_row):

            ax.axhspan(0, cost,color='green',alpha=0.2)
            ax.boxplot(
                    data,
                    medianprops=dict(color='red', linewidth=2),
                    flierprops=dict(
                        marker='_',        
                        markersize=8,
                        markeredgecolor='black',
                        markeredgewidth=0.5
                    )
                )
            for x, (p25, med, p75) in enumerate(prec_stats, start=1):
                ax.fill_between([x - 0.25, x + 0.25],[p25, p25],[p75, p75],color='blue',alpha=0.4,zorder=1)
                ax.plot([x - 0.25, x + 0.25],[med, med],color='blue',linewidth=2,zorder=2)

            ax.set_xticks(list(range(1,len(list_n_ep)+1)))
            ax.set_xticklabels([str(cost) for cost in list_n_ep])

            if first_row:
                ax.set_title(abbreviate(pack.replace('pack_PPO_','')),fontsize=8)
            if not first_column:
                ax.set_yticks([])
            if first_column:
                ax.set_ylabel(r'$c_{val}/c_{learn}$')
                ax.text(-0.5, 0.5,rf'$\varphi_c={cost}$',transform=ax.transAxes,ha='center',va='center',rotation=90)
            if not last_row:
                ax.set_xticks([])
            if last_row:
                ax.set_xlabel(r'$\kappa$')

        fig, axs = plt.subplots(len(list_cost),len(self.all_packs), figsize=(1.8*len(self.all_packs),1.5*len(list_cost)))
        plt.subplots_adjust(top=0.95,bottom=0.1,left=0.07,right=0.98, hspace=0.1,wspace=0.1)

        for i,pack in tqdm(enumerate(self.all_packs)):
            first_column=[True if i==0 else False][0]

            df_costs=pd.read_csv(data_path+'/'+pack.replace('pack','SB3')+'/df_test_cost.csv')
            df_precs=pd.read_csv(data_path+'/'+pack.replace('pack','SB3')+'/df_test_prec.csv')

            
            for j,cost in enumerate(list_cost):
                first_row=[True if j==0 else False][0]
                last_row=[True if j==len(list_cost)-1 else False][0]

                # Guardar todos los ratios de coste para despues dibujar boxplot
                all_ratios=[]
                all_precs=[]
                for k,n_ep in enumerate(list_n_ep):
                    df_cost_current=df_costs[[c for c in df_costs.columns if f"_{n_ep}_{cost}cost_" in c]]
                    all_ratios.append([1 if ratio > 1 else ratio for ratio in df_cost_current.values.ravel().tolist()])

                    df_prec_current=df_precs[[c for c in df_precs.columns if f"_{n_ep}_{cost}cost_" in c]]
                    list_prec=df_prec_current.values.ravel().tolist()
                    all_precs.append([np.percentile(list_prec,25),np.median(list_prec),np.percentile(list_prec,75)])


                # Dibujar grafica de boxplots por del pack-threshold actual
                plot_cost_boxplots_per_pack_threshold(axs[j,i],all_ratios,all_precs,list_n_ep,pack,cost,first_column,first_row,last_row)

        plt.savefig(self.figure_path+'/all_setups/appendix_cost_of_cost_driven_test_boxplots_pack_cost.png')


        # GRFACICA 4: boxplot por pack-periodo-n_ep junto a IQR-median de prec para default test criterion
        def plot_cost_boxplots_per_pack_freq(ax,data,prec_stats,list_n_ep,pack,freq,first_column,first_row,last_row):

            ax.axhspan(0, 0.2,color='green',alpha=0.2)
            ax.boxplot(
                    data,
                    medianprops=dict(color='red', linewidth=2),
                    flierprops=dict(
                        marker='_',        
                        markersize=8,
                        markeredgecolor='black',
                        markeredgewidth=0.5
                    )
                )
            for x, (p25, med, p75) in enumerate(prec_stats, start=1):
                ax.fill_between([x - 0.25, x + 0.25],[p25, p25],[p75, p75],color='blue',alpha=0.4,zorder=1)
                ax.plot([x - 0.25, x + 0.25],[med, med],color='blue',linewidth=2,zorder=2)
            ax.set_xticks(list(range(1,len(list_n_ep)+1)))
            ax.set_xticklabels([str(cost) for cost in list_n_ep])

            if first_row:
                ax.set_title(abbreviate(pack.replace('pack_PPO_','')),fontsize=8)
            if not first_column:
                ax.set_yticks([])
            if first_column:
                ax.set_ylabel(r'$c_{val}/c_{learn}$')
                ax.text(-0.5, 0.5,rf'$\varphi={freq}$',transform=ax.transAxes,ha='center',va='center',rotation=90)
            if not last_row:
                ax.set_xticks([])
            if last_row:
                ax.set_xlabel(r'$\kappa$')

        list_freq=[20,10,5,1]

        fig, axs = plt.subplots(len(list_freq),len(self.all_packs), figsize=(1.8*len(self.all_packs),1.5*len(list_freq)))
        plt.subplots_adjust(top=0.95,bottom=0.1,left=0.07,right=0.98, hspace=0.1,wspace=0.1)

        for i,pack in tqdm(enumerate(self.all_packs)):
            first_column=[True if i==0 else False][0]

            df_costs=pd.read_csv(data_path+'/'+pack.replace('pack','SB3')+'/df_test_cost.csv')
            df_precs=pd.read_csv(data_path+'/'+pack.replace('pack','SB3')+'/df_test_prec.csv')

            
            for j,freq in enumerate(list_freq):
                first_row=[True if j==0 else False][0]
                last_row=[True if j==len(list_freq)-1 else False][0]

                # Guardar todos los ratios de coste para despues dibujar boxplot
                all_ratios=[]
                all_precs=[]
                for k,n_ep in enumerate(list_n_ep):
                    df_current=df_costs[[c for c in df_costs.columns if f"_{n_ep}_{freq}_" in c]]
                    all_ratios.append([1 if ratio > 1 else ratio for ratio in df_current.values.ravel().tolist()])

                    df_prec_current=df_precs[[c for c in df_precs.columns if f"_{n_ep}_{freq}_" in c]]
                    list_prec=df_prec_current.values.ravel().tolist()
                    all_precs.append([np.percentile(list_prec,25),np.median(list_prec),np.percentile(list_prec,75)])


                # Dibujar grafica de boxplots por del pack-threshold actual
                plot_cost_boxplots_per_pack_freq(axs[j,i],all_ratios,all_precs,list_n_ep,pack,freq,first_column,first_row,last_row)

        plt.savefig(self.figure_path+'/all_setups/appendix_cost_of_cost_driven_test_boxplots_pack_freq.png')

    def val_freq_throughout_sequence(self):

        def plot_val_freq_evolution(ax,df,pack,last_row):

            img = (df.to_numpy().T > 0).astype(int)
            cmap = ListedColormap(["white", "black"])
            ax.imshow(img,cmap=cmap,aspect='auto',vmin=0,vmax=1,interpolation='nearest')
            ax.set_yticks([])
            ax.set_ylabel(abbreviate(pack.replace('pack_PPO_','')),fontsize=8)
            if last_row:
                ax.set_xlabel(r'$t$')

            ax.tick_params(axis='x', labelsize=8,length=0.5)


        fig, axs = plt.subplots(len(self.all_packs),1, figsize=(5,1*len(self.all_packs)))
        plt.subplots_adjust(top=0.99,bottom=0.05,left=0.05,right=0.98, hspace=0.25,wspace=0.2)


        df_conf=pd.read_csv(self.data_path+'/configurations.csv')
        test_conf=df_conf.loc[df_conf["pack"] == 'all', "test_cost_freq_opt"].iloc[0]
        n_ep=test_conf.split('_')[0]
        cost=test_conf.split('_')[1]

        for i,pack in enumerate(self.all_packs):
            last_row=[True if i==len(self.all_packs)-1 else False][0]

            # Columnas de la base de datos de interes
            df_n_ep=pd.read_csv(data_path+'/'+pack.replace('pack','SB3')+'/df_test_n_ep.csv')
            df_current=df_n_ep[[c for c in df_n_ep.columns if f"_{n_ep}_{cost}cost_" in c]]

            # Dibujar grafica por pack de las 30 semillas
            
            plot_val_freq_evolution(axs[i],df_current,pack,last_row)


        plt.savefig(self.figure_path+'/all_setups/appendix_val_freq_throughout_sequence.pdf')

    def resource_allocation(self):
        pass

grapher=Grapher(library,all_packs,seeds,data_path,figure_path) 

grapher.deg_with_without_norm(packs_with_norm)
grapher.estimations()
grapher.estimations(train_white=True)
grapher.regions_with_deg()
grapher.criteria_configurations()
grapher.val_cost_throughout_sequence()
grapher.cost_of_cost_driven_test()
grapher.val_freq_throughout_sequence()
