from Main import *
import seaborn as sns
from matplotlib.patches import Rectangle

# Common variables
data_path='experiments/results/data'
figure_path='experiments/results/figures'
library='SB3'
global_deg_metric='norm_from_mean_worsening_to_improvement'
local_deg_metric='reward_diff'
prec_metric='relative_perc_criteria_best'
eff_metric='first_time_to_same_reward'

seeds=list(range(1,31))
all_packs=[ # ClassicControl
            #'pack_PPO_Pendulum',
            # Box2D
            'pack_PPO_LunarLanderContinuous',
            'pack_PPO_BipedalWalker',
            # MuJoCo
            'pack_PPO_Swimmer',
            'pack_PPO_Hopper',
            'pack_PPO_Ant',
            'pack_PPO_HalfCheetah',
            'pack_PPO_Walker2d'        
                            ]


# Common functions
def abbreviate(env_name: str) -> str:
    NAME_TO_ABBR = {
    #'Pendulum': 'P',
    'LunarLanderContinuous': 'LLC',
    'BipedalWalker': 'BW',
    'Ant': 'A',
    'Hopper': 'H',
    'HalfCheetah': 'HC',
    'Walker2d': 'W2d',
    'Swimmer': 'S'
}
    return NAME_TO_ABBR.get(env_name, env_name)


# Class for generating figures of the main analysis
class Grapher():

    def __init__(self,library,all_packs,seeds,data_path,figure_path):
        self.library=library
        self.all_packs=all_packs
        self.seeds=seeds
        self.data_path=data_path
        self.figure_path=figure_path

    def deg_prec_eff(self):

        # Funcion para extraer datos de interes
        data_deg=[[0,0,0] for _ in range(4)]
        data_acc_last,data_acc_train,data_acc_test=[[0,0,0] for _ in range(4)],[[0,0,0] for _ in range(4)],[[0,0,0] for _ in range(4)]
        data_eff_last,data_eff_train,data_eff_test=[[0,0,0] for _ in range(4)],[[0,0,0] for _ in range(4)],[[0,0,0] for _ in range(4)]

        def from_metric_data_to_count(data1,data2,data3,count_list):

            m1,m2,m3=np.median(data1),np.median(data2),np.median(data3)

            for i,m in enumerate([m1,m2,m3]):

                if m<0.25:
                    count_list[i][0]+=1
                if m>=0.25 and m<0.75:
                    count_list[i][1]+=1
                if m>=0.75:
                    count_list[i][2]+=1
            
            for i in range(3):
                count_list[3][i]=count_list[0][i]+count_list[1][i]+count_list[2][i]

            return count_list

        # Funciones para plotear datos pack-region
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
                        Line2D([0], [0], marker='s', color='green', label='Validation', markersize=6, linestyle='')
                    ]
                    ax.legend(handles=legend_elements, loc='upper center',bbox_to_anchor=(1, -0.4), ncol=3, frameon=False)
            
        # Cargar datos necesarios
        df_conf=pd.read_csv(self.data_path+'/configurations.csv')
        train_conf=str(int(df_conf.loc[df_conf["pack"] == 'all', "train_opt"].iloc[0]))
        conf_str=df_conf.loc[df_conf["pack"] == 'all', "test_cost_freq_opt"].iloc[0]
        test_conf=conf_str.split('_')[0]+'_'+conf_str.split('_')[1]+'cost'
            
        # Graficas
        fig, axs = plt.subplots(len(self.all_packs),3*3+2, figsize=(5*3,0.65*len(self.all_packs)),
                                    gridspec_kw={'width_ratios': [1,1,1,0.35,1,1,1,0.35,1,1,1]})
        plt.subplots_adjust(top=0.9,bottom=0.15,left=0.05,right=0.98, hspace=0.0,wspace=0.0)

        #---- Degradacion
        for i,pack in enumerate(all_packs):

            pack_path=self.data_path+'/'+pack.replace('pack',self.library)+'/'

            # Obtener los datos de degradacion por region de este pack
            deg1,deg2,deg3=DataConverter.from_df_data_to_graph_data(
                                    [pack_path+'learning_regions.csv',pack_path+'deg_evolution.csv'],pack,'deg_distribution',
                                    global_deg_metric=global_deg_metric,local_deg_metric=local_deg_metric
                                ) 

            # Dibujar distribucion de degradacion de las tres regiones
            title=[['\nInitialization','Degradation\nLearning','\nConvergence']if i==0 else ['']*3][0]
            not_last_pack=[False if i==len(all_packs)-1 else True][0]
            plot_pack_degradation(axs[i,0],deg1,pack_name=pack.replace('pack_PPO_',''),title=title[0],not_last_pack=not_last_pack)
            plot_pack_degradation(axs[i,1],deg2,title=title[1],not_last_pack=not_last_pack)
            plot_pack_degradation(axs[i,2],deg3,title=title[2],not_last_pack=not_last_pack)

            # Actualizar datos de interes
            data_deg=from_metric_data_to_count(deg1,deg2,deg3,data_deg)

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
            title=[['\nInitialization','Accuracy\nLearning','\nConvergence']if i==0 else ['']*3][0]
            last_pack=[True if i==len(all_packs)-1 else False][0]
            plot_pack_criteria_prec_or_eff(axs[i,4],[last1,train1,test1][::-1],pack_name=pack.replace('pack_PPO_',''),title=title[0],last_pack=last_pack)
            plot_pack_criteria_prec_or_eff(axs[i,5],[last2,train2,test2][::-1],title=title[1],last_pack=last_pack)
            plot_pack_criteria_prec_or_eff(axs[i,6],[last3,train3,test3][::-1],title=title[2],last_pack=last_pack)

            # Actualizar datos de interes
            data_acc_last=from_metric_data_to_count(last1,last2,last3,data_acc_last)
            data_acc_train=from_metric_data_to_count(train1,train2,train3,data_acc_train)
            data_acc_test=from_metric_data_to_count(test1,test2,test3,data_acc_test)

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
            title=[['\nInitialization','Efficiency\nLearning','\nConvergence']if i==0 else ['']*3][0]
            last_pack=[True if i==len(all_packs)-1 else False][0]
            plot_pack_criteria_prec_or_eff(axs[i,8],[last1,train1,test1][::-1],pack_name=pack.replace('pack_PPO_',''),title=title[0],last_pack=last_pack)
            plot_pack_criteria_prec_or_eff(axs[i,9],[last2,train2,test2][::-1],title=title[1],last_pack=last_pack)
            plot_pack_criteria_prec_or_eff(axs[i,10],[last3,train3,test3][::-1],title=title[2],last_pack=last_pack)

            # Actualizar datos de interes
            data_eff_last=from_metric_data_to_count(last1,last2,last3,data_eff_last)
            data_eff_train=from_metric_data_to_count(train1,train2,train3,data_eff_train)
            data_eff_test=from_metric_data_to_count(test1,test2,test3,data_eff_test)


        for i in range(len(all_packs)):
                axs[i,3].axis('off')
                axs[i,7].axis('off')

        # Guardar base de datos de interes
        all_data=[data_deg,data_acc_last,data_acc_train,data_acc_test,
                  data_eff_last,data_eff_train,data_eff_test]
        metric_criteria=[['deg',None],['acc','last'],['acc','train'],['acc','test'],
                         ['eff','last'],['eff','train'],['eff','test']]
        columns=['metric','criteria',
                                   'r1_init','r2_init','r3_init',
                                   'r1_learn','r2_learn','r3_learn',
                                   'r1_conv','r2_conv','r3_conv',
                                   'r1_all','r2_all','r3_all']
        
        #---- Porcentajes
        df = pd.DataFrame(columns=columns)
        cases=len(self.all_packs)
        for case, data in zip(metric_criteria,all_data):

            df.loc[len(df)] = case+ [j/cases for i in data[:-1] for j in i]+[i/(cases*3) for i in data[-1]]

        # Todos los criterios para acc 
        cases=len(self.all_packs)*3
        data=np.array([j for i in data_acc_last for j in i])+np.array([j for i in data_acc_train for j in i])+np.array([j for i in data_acc_test for j in i])
        df.loc[len(df)] = ['acc','all']+ [i/cases for i in data[:-3]]+[i/(cases*3) for i in data[-3:]]

        # Todos los criterios para eff
        data=np.array([j for i in data_eff_last for j in i])+np.array([j for i in data_eff_train for j in i])+np.array([j for i in data_eff_test for j in i])
        df.loc[len(df)] = ['eff','all']+ [i/cases for i in data[:-3]]+[i/(cases*3) for i in data[-3:]]

        df.to_csv(self.data_path+'/paper/df_deg_acc_eff.csv', index=False)

        # Para paper solo columnas de alls
        df_all = df.iloc[:, [0, 1] + list(range(-3, 0))]
        df_all.columns = ['Metric', 'Criterion', 'Low', 'Middle', 'High']

        df_all.to_latex(self.data_path+'/paper/df_deg_acc_eff.tex',
                index=False,
                escape=False,
                float_format="%.2f",
                column_format="|c|c|c|c|c|"
            )



        #---- Fracciones
        df = pd.DataFrame(columns=columns)
        
        cases=len(self.all_packs)
        for case, data in zip(metric_criteria,all_data):

            df.loc[len(df)] = case+ [str(j)+'/'+str(cases) for i in data[:-1] for j in i]+[str(i)+'/'+str(cases*3) for i in data[-1]]

        # Todos los criterios para acc 
        cases=len(self.all_packs)*3
        data=np.array([j for i in data_acc_last for j in i])+np.array([j for i in data_acc_train for j in i])+np.array([j for i in data_acc_test for j in i])
        df.loc[len(df)] = ['acc','all']+ [str(i)+'/'+str(cases) for i in data[:-3]]+[str(i)+'/'+str(cases*3) for i in data[-3:]]

        # Todos los criterios para eff
        data=np.array([j for i in data_eff_last for j in i])+np.array([j for i in data_eff_train for j in i])+np.array([j for i in data_eff_test for j in i])
        df.loc[len(df)] = ['eff','all']+ [str(i)+'/'+str(cases) for i in data[:-3]]+[str(i)+'/'+str(cases*3) for i in data[-3:]]

        df.to_csv(self.data_path+'/paper/df_deg_acc_eff_frac.csv', index=False)

        # Para paper solo columnas de alls
        df = pd.DataFrame(columns=['Metric', 'Criterion', 'Low', 'Middle','High','Total'])

        cases=len(self.all_packs)
        for case, data in zip(metric_criteria,all_data):
            df.loc[len(df)] = case+ [i for i in data[-1]]+[cases*3]
        
        cases=len(self.all_packs)*3
        data=np.array([j for i in data_acc_last for j in i])+np.array([j for i in data_acc_train for j in i])+np.array([j for i in data_acc_test for j in i])
        df.loc[len(df)] = ['acc','all']+[i for i in data[-3:]]+[cases*3]
        data=np.array([j for i in data_eff_last for j in i])+np.array([j for i in data_eff_train for j in i])+np.array([j for i in data_eff_test for j in i])
        df.loc[len(df)] = ['eff','all']+[i for i in data[-3:]]+[cases*3]


        df.to_latex(self.data_path+'/paper/df_deg_acc_eff_all.tex',
                index=False,
                escape=False,
                float_format="%.2f",
                column_format="|c|c|c|c|c|"
            )





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
                "test":  Line2D([0], [0], marker='s', color='green',   label='Validation', markersize=6, linestyle='')
            }

        legend_groups = [["last", "train"],["last", "test"],["train", "test"]]

        # Funcion para guardar datos de interes
        if comparison_type=='how_times_best':
            last_train,last_test,train_test=[[0,0,0] for _ in range(3)],[[0,0,0] for _ in range(3)],[[0,0,0] for _ in range(3)]
        if comparison_type=='in_which_deg_best':
            last_train,last_test,train_test=[[[0,0,0] for _ in range(4)] for _ in range(2)],[[[0,0,0] for _ in range(4)] for _ in range(2)],[[[0,0,0] for _ in range(4)] for _ in range(2)]

        def from_data_times_to_mean_perc(data1,data2,data3,last_train,last_test,train_test):
            
            for i,data in enumerate([data1,data2,data3]):


                last_train[i][0]+=data[0][0]
                last_train[i][2]+=data[0][1]
                last_train[i][1]+=1-data[0][0]-data[0][1]

                last_test[i][0]+=data[1][0]
                last_test[i][2]+=data[1][1]
                last_test[i][1]+=1-data[1][0]-data[1][1]

                train_test[i][0]+=data[2][0]
                train_test[i][2]+=data[2][1]
                train_test[i][1]+=1-data[2][0]-data[2][1]


            return last_train,last_test,train_test

        def from_data_conditioned_deg_to_perc(data1,data2,data3,last_train,last_test,train_test):

            for i,deg_pair in enumerate([data1,data2,data3]):

                #<0.25
                last_train[0][i][0]+= sum(d < 0.25 for d in deg_pair[0][0])
                last_train[1][i][0]+= sum(d < 0.25 for d in deg_pair[0][1])
                last_test[0][i][0]+= sum(d < 0.25 for d in deg_pair[1][0])
                last_test[1][i][0]+= sum(d < 0.25 for d in deg_pair[1][1])
                train_test[0][i][0]+= sum(d < 0.25 for d in deg_pair[2][0])
                train_test[1][i][0]+= sum(d < 0.25 for d in deg_pair[2][1])

                #>=0.25 y <0.75
                last_train[0][i][1]+= sum(d >= 0.25 and d<0.75 for d in deg_pair[0][0])
                last_train[1][i][1]+= sum(d >= 0.25 and d<0.75 for d in deg_pair[0][1])
                last_test[0][i][1]+= sum(d >= 0.25 and d<0.75 for d in deg_pair[1][0])
                last_test[1][i][1]+= sum(d >= 0.25 and d<0.75 for d in deg_pair[1][1])
                train_test[0][i][1]+= sum(d >= 0.25 and d<0.75 for d in deg_pair[2][0])
                train_test[1][i][1]+= sum(d >= 0.25 and d<0.75 for d in deg_pair[2][1])

                #>=0.75
                last_train[0][i][2]+= sum(d >=0.75 for d in deg_pair[0][0])
                last_train[1][i][2]+= sum(d >=0.75 for d in deg_pair[0][1])
                last_test[0][i][2]+= sum(d >=0.75 for d in deg_pair[1][0])
                last_test[1][i][2]+= sum(d >=0.75 for d in deg_pair[1][1])
                train_test[0][i][2]+= sum(d >=0.75 for d in deg_pair[2][0])
                train_test[1][i][2]+= sum(d >=0.75 for d in deg_pair[2][1])

            for i in range(3):

                last_train[0][3][i]=last_train[0][0][i]+last_train[0][1][i]+last_train[0][2][i]
                last_train[1][3][i]=last_train[1][0][i]+last_train[1][1][i]+last_train[1][2][i]
                last_test[0][3][i]=last_test[0][0][i]+last_test[0][1][i]+last_test[0][2][i]
                last_test[1][3][i]=last_test[1][0][i]+last_test[1][1][i]+last_test[1][2][i]
                train_test[0][3][i]=train_test[0][0][i]+train_test[0][1][i]+train_test[0][2][i]
                train_test[1][3][i]=train_test[1][0][i]+train_test[1][1][i]+train_test[1][2][i]

            return last_train,last_test,train_test


        # Funciones para plotear datos por pack-region
        def plot_how_many_times_better(axs, data, pack_name='', title='', not_last_pack=True):

            color_marker_map = {'green': 's','orange': '^','blue': 'o'}
            colors = [['blue', 'orange'],['blue', 'green'],['orange', 'green']]

            # recorrer cada región → cada ax
            for i, (ax, (pair, (color_left, color_right))) in enumerate(zip(axs, zip(data, colors))):

                left_val, right_val = pair

                # barras enfrentadas
                ax.barh(
                    y=0,
                    width=left_val,
                    left=0,
                    color=color_left,
                    height=0.8,alpha=0.9
                )

                ax.barh(
                    y=0,
                    width=right_val,
                    left=1 - right_val,
                    color=color_right,
                    height=0.8,alpha=0.9
                )

                for x in [0.25, 0.5, 0.75]:
                    ax.axvline(x, color='black', linestyle='--', linewidth=0.5)

                # marcadores extremos
                marker_left = color_marker_map.get(color_left, 'o')
                marker_right = color_marker_map.get(color_right, 'o')

                if left_val > 0:
                    ax.plot(0, 0, marker=marker_left, color=color_left, markersize=5,markeredgecolor='black',)

                if right_val > 0:
                    ax.plot(1, 0, marker=marker_right, color=color_right, markersize=5,markeredgecolor='black')

                # estética por ax
                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(-0.5, 0.5)
                ax.set_yticks([])
                ax.set_ylabel(abbreviate(pack_name), rotation=0, labelpad=20)
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
                ax.set_xticklabels(['0', '0.25', '0.5', '0.75', '1'])
                

                #if not_last_pack:
                ax.set_xticks([])

            axs[0].set_title(title)
            # leyenda solo en último
            if not not_last_pack and pack_name != '':
                for i in range(3):
                        handles = [legend_map[k] for k in legend_groups[i]]
                        axs[i].legend(handles=handles,loc='upper center',bbox_to_anchor=(0.5, 0.5),ncol=len(handles),frameon=False)
                        

            # etiqueta global (opcional)
            for i in range(3):
                axs[i].set_ylabel(abbreviate(pack_name),rotation=0, labelpad=20, fontsize=8)
                axs[i].yaxis.set_label_coords(-0.15, 0.1)

        # NOTE: para sugerencia de Etor descomentar
        # def plot_how_many_times_better(axs, data, pack_name='', title='', not_last_pack=True):


        #     mpl.rcParams['hatch.linewidth'] = 0.5

        #     colors = [['blue', 'orange'],['blue', 'green'],['orange', 'green']]
        #     hatch_map = {'blue': '..','orange': '//', 'green': '||'}

        #     # --- PLOTS ---
        #     for i, (ax, (pair, (color_left, color_right))) in enumerate(zip(axs, zip(data, colors))):

        #         left_val, right_val = pair

        #         ax.barh(
        #             y=0,
        #             width=left_val,
        #             left=0,
        #             color=color_left,
        #             hatch=hatch_map[color_left],
        #             edgecolor='black',
        #             linewidth=0,
        #             height=0.8
        #         )

        #         ax.barh(
        #             y=0,
        #             width=right_val,
        #             left=1-right_val,
        #             color=color_right,
        #             hatch=hatch_map[color_right],
        #             edgecolor='black',
        #             linewidth=0,
        #             height=0.8
        #         )

        #         ax.set_xlim(-0.05, 1.05)
        #         ax.set_ylim(-0.5, 0.5)
        #         ax.set_yticks([])
        #         ax.set_xticks([])

        #     axs[0].set_title(title)

        #     # --- etiquetas globales ---
        #     for i in range(3):
        #         axs[i].set_ylabel(abbreviate(pack_name),rotation=0,labelpad=20,fontsize=8)
        #         axs[i].yaxis.set_label_coords(-0.15, 0.1)

        # NOTE: esta seguramente para borra, hace normalizacion individual y da lugar a sesgos
        # def plot_in_which_deg_best(axs, data, pack_name='', title='', not_last_pack=True):

        #     colors = [
        #         ['blue', 'orange'],
        #         ['blue', 'green'],
        #         ['orange', 'green']
        #     ]

        #     color_marker_map = {
        #         'blue': 'o',
        #         'orange': '^',
        #         'green': 's'
        #     }

        #     bins = 10
        #     y_spacing = 0.5
        #     hist_height = 0.4

        #     # recorrer 3 regiones → 3 axes
        #     for i, (ax, pair, pair_colors) in enumerate(zip(axs, data, colors)):

        #         bar_height = hist_height / len(pair_colors)

        #         for j, (sublist, color) in enumerate(zip(pair, pair_colors)):

        #             if len(sublist) == 0:
        #                 continue

        #             sublist = np.asarray(sublist)

        #             # histograma
        #             hist, bin_edges = np.histogram(
        #                 sublist, bins=bins, range=(0, 1), density=False
        #             )

        #             # normalización
        #             percentages = hist / hist.sum() if hist.sum() > 0 else hist

        #             bin_width = bin_edges[1] - bin_edges[0]

        #             # color degradado
        #             base_rgb = np.array(mcolors.to_rgb(color))
        #             colors_scaled = [
        #                 tuple(base_rgb * p + (1 - p)) for p in percentages
        #             ]

        #             # posición vertical dentro del ax
        #             bottom = j * bar_height

        #             for left, col in zip(bin_edges[:-1], colors_scaled):
        #                 ax.bar(
        #                     left,
        #                     bar_height,
        #                     width=bin_width,
        #                     bottom=bottom,
        #                     align='edge',
        #                     color=col,
        #                     edgecolor=None
        #                 )

        #             # mediana
        #             median_val = np.median(sublist)
        #             mid_y = bottom + bar_height / 2
        #             marker = color_marker_map.get(color, 'o')

        #             ax.plot(median_val, mid_y, marker=marker,
        #                     color='black', markersize=3)

        #         # estética por ax
        #         ax.set_xlim(-0.05, 1.05)
        #         ax.set_yticks([])
        #         ax.set_title(title)
        #         ax.set_ylabel(abbreviate(pack_name), rotation=0, labelpad=20)
        #         ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        #         ax.set_xticklabels(['0', '0.25', '0.5', '0.75', '1'])

        #         if not_last_pack:
        #             ax.set_xticklabels([])

        #     # leyenda solo en el último si aplica
        #     if not_last_pack:
        #         for ax in axs:
        #             ax.set_xticklabels([])
        #     else:
        #         if pack_name != '':
        #             for i in range(3):
        #                 handles = [legend_map[k] for k in legend_groups[i]]
        #                 axs[i].legend(handles=handles,loc='upper center',bbox_to_anchor=(0.5, -0.4),ncol=len(handles),frameon=False)

        # NOTE: Nuevo con normalizacion conjunta, para evitar el sesgo que nombra Aritz
        def plot_in_which_deg_best(axs, data, pack_name='', title='', not_last_pack=True):

            colors = [['blue', 'orange'],['blue', 'green'],['orange', 'green']]
            color_marker_map = { 'blue': 'o','orange': '^','green': 's'}

            bins = 10
            hist_height = 0.4

            # recorrer 3 regiones → 3 axes
            for i, (ax, pair, pair_colors) in enumerate(zip(axs, data, colors)):

                bar_height = hist_height / len(pair_colors)

                # NUEVO: calcular histogramas PRIMERO (sin normalizar)
                hist_list = []
                for sublist in pair:
                    sublist = np.asarray(sublist)
                    hist, bin_edges = np.histogram(
                        sublist, bins=bins, range=(0, 1), density=False
                    )
                    hist_list.append(hist)

                # NUEVO: normalización conjunta dentro del pair
                pair_max = max([h.max() if len(h) > 0 else 0 for h in hist_list])
                pair_max = pair_max if pair_max > 0 else 1
                norm_hist_list = [h / pair_max for h in hist_list]
                bin_width = bin_edges[1] - bin_edges[0]


                # Grafica
                for j, (hist, color) in enumerate(zip(norm_hist_list, pair_colors)):

                    # degradado consistente entre las dos sublistas del pair
                    base_rgb = np.array(mcolors.to_rgb(color))
                    colors_scaled = [tuple(base_rgb * p + (1 - p)) for p in hist]
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
                    if len(pair[j]) > 0:
                        median_val = np.median(pair[j])
                        mid_y = bottom + bar_height / 2
                        marker = color_marker_map.get(color, 'o')

                        ax.plot(
                            median_val,
                            mid_y,
                            marker=marker,
                            color='black',
                            markersize=3
                        )

                # estética por ax
                ax.set_xlim(-0.05, 1.05)
                ax.set_yticks([])
                ax.set_title(title)
                ax.set_ylabel(abbreviate(pack_name), rotation=0, labelpad=20)
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
                ax.set_xticklabels(['0', '0.25', '0.5', '0.75', '1'])

                if not_last_pack:
                    ax.set_xticklabels([])

            # leyenda
            if not_last_pack:
                for ax in axs:
                    ax.set_xticklabels([])
            else:
                if pack_name != '':
                    for i in range(3):
                        handles = [legend_map[k] for k in legend_groups[i]]
                        axs[i].legend(
                            handles=handles,
                            loc='upper center',
                            bbox_to_anchor=(0.5, -0.4),
                            ncol=len(handles),
                            frameon=False
                        )

        # NOTE: Nuevo con barplots enfrentadas por nivel de degradacion, relacionado con propuesta de Aritz
        def plot_times_best_by_deg(axs, data, pack_name='', title='', not_last_pack=True):


            colors = [
                ['blue', 'orange'],
                ['blue', 'green'],
                ['orange', 'green']
            ]

            color_map = {
                'blue': 'blue',
                'orange': 'orange',
                'green': 'green'
            }

            color_marker_map = {
                'blue': 'o',
                'orange': '^',
                'green': 's'
            }

            n_bins = 10
            bar_width = 1

            # =========================================================
            # recorrer 3 panels
            # =========================================================
            for i, (ax, pair, pair_colors) in enumerate(zip(axs, data, colors)):

                pair = np.array(pair)  # shape (8, 2)

                # normalización global por bin (para comparar alturas)
                bin_sums = pair.sum(axis=1, keepdims=True)
                bin_sums[bin_sums == 0] = 1

                norm_pair = pair / bin_sums

                # =========================================================
                # dibujo por bin
                # =========================================================
                x_positions = np.arange(n_bins)

                for b in range(n_bins):

                    val_a, val_b = norm_pair[b]

                    c1, c2 = pair_colors

                    # =====================================================
                    # barra inferior (primer elemento)
                    # =====================================================
                    ax.bar(
                        x_positions[b],
                        val_a,
                        width=bar_width,
                        bottom=0,
                        color=c1,
                        alpha=0.7,
                        edgecolor=None
                    )

                    # =====================================================
                    # barra superior (segundo elemento)
                    # =====================================================
                    ax.bar(
                        x_positions[b],
                        val_b,
                        width=bar_width,
                        bottom=1 - val_b,
                        color=c2,
                        alpha=0.7,
                        edgecolor=None
                    )

                # =========================================================
                # estética
                # =========================================================
                ax.set_ylim(0, 1)
                ax.set_yticks([])
                ax.set_yticklabels([])


                ax.set_xticks(np.arange(n_bins))
                ax.set_xticks([-0.5, 2, 4.5, 7, 9.5])
                ax.set_xticklabels(['0', '0.25', '0.5', '0.75', '1'])

                ax.set_title(title)
                ax.set_ylabel(abbreviate(pack_name), rotation=0, labelpad=20)

                if not_last_pack:
                    ax.set_xticklabels([])

            # =========================================================
            # leyenda / limpieza
            # =========================================================
            if not_last_pack:
                for ax in axs:
                    ax.set_xticklabels([])
            else:
                if pack_name != '':
                    for i in range(3):
                        handles = [legend_map[k] for k in legend_groups[i]]
                        axs[i].legend(
                            handles=handles,
                            loc='upper center',
                            bbox_to_anchor=(0.5, -0.4),
                            ncol=len(handles),
                            frameon=False
                        )

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
        if comparison_type=='how_times_best':
            # NOTE: para sugerencia de Etor sustituit los [2] por [0.5]
            fig, axs = plt.subplots(len(self.all_packs)*3+2,3, figsize=(5,4.8*0.1*len(self.all_packs)),
                                        gridspec_kw={'height_ratios': [1]*len(self.all_packs)+[2]+[1]*len(self.all_packs)+[2]+[1]*len(self.all_packs)})
            plt.subplots_adjust(top=0.92,bottom=0.05,left=0.1,right=0.98, hspace=0,wspace=0.0)
            #plt.subplots_adjust(top=0.92,bottom=0.1,left=0.1,right=0.98, hspace=0.0,wspace=0.0) # NOTE: para sugerencia de Etor descomentar
        else:
            fig, axs = plt.subplots(len(self.all_packs),3*3+2, figsize=(5*3,0.6*len(self.all_packs)),
                                        gridspec_kw={'width_ratios': [1,1,1,0.35,1,1,1,0.35,1,1,1]})
            plt.subplots_adjust(top=0.95,bottom=0.15,left=0.05,right=0.98, hspace=0.0,wspace=0.0)

        for i,pack in enumerate(all_packs):

            pack_path=self.data_path+'/'+pack.replace('pack',self.library)+'/'
            title=[['Initialization','Learning','Convergence']if i==0 else ['']*3][0]
            not_last_pack=[False if i==len(all_packs)-1 else True][0]

            p=len(self.all_packs)
            
            if comparison_type=='how_times_best':

                # Obtener los datos de la medida de comapracion por region de este pack            
                matrix1,matrix2,matrix3,_,_,_=DataConverter.from_df_data_to_graph_data(
                                        [pack_path+'learning_regions.csv',pack_path+'df_last_truth.csv',
                                        pack_path+'df_train_truth.csv',pack_path+'df_test_truth.csv'],
                                        pack,which_graph=comparison_type,train_conf=train_conf,test_conf=test_conf)
                # Dibujar distribucion de la medida de comparacion de las tres regiones
                plot_how_many_times_better([axs[i,0],axs[p+i+1,0],axs[p*2+i+2,0]],matrix1,pack_name=pack.replace('pack_PPO_',''),title=title[0],not_last_pack=not_last_pack)
                plot_how_many_times_better([axs[i,1],axs[p+i+1,1],axs[p*2+i+2,1]],matrix2,title=title[1],not_last_pack=not_last_pack)
                plot_how_many_times_better([axs[i,2],axs[p+i+1,2],axs[p*2+i+2,2]],matrix3,title=title[2],not_last_pack=not_last_pack)

                # Datos de interes
                last_train,last_test,train_test=from_data_times_to_mean_perc(matrix1,matrix2,matrix3,last_train,last_test,train_test)

                # NOTE: para sugerencia de Etor descomentar
                # legend_handles = [
                #     mpatches.Patch(facecolor='blue', edgecolor='black', hatch='..', label='last'),
                #     mpatches.Patch(facecolor='orange', edgecolor='black', hatch='//', label='train'),
                #     mpatches.Patch(facecolor='green', edgecolor='black', hatch='||', label='test'),
                #     mpatches.Patch(facecolor='white', edgecolor='black', hatch=None, label='ties')
                # ]
                # axs[p*2+p+1,0].legend(
                #                     handles=legend_handles,
                #                     loc='lower center',
                #                     ncol=4,
                #                     frameon=False,
                #                     bbox_to_anchor=(1.2, -3.6),
                #                     borderpad=1.2  
                #                 )

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

                #Datos de interes
                last_train,last_test,train_test=from_data_conditioned_deg_to_perc(matrix1,matrix2,matrix3,last_train,last_test,train_test)

            # NOTE: nuevo para la sugerencia de Aritz de p(A wins B|deg)
            if comparison_type=='how_times_best_in_deg_level':
                matrix1,matrix2,matrix3=DataConverter.from_df_data_to_graph_data(
                                [pack_path+'learning_regions.csv',pack_path+'df_last_truth.csv',
                                pack_path+'df_train_truth.csv',pack_path+'df_test_truth.csv',pack_path+'deg_evolution.csv'],
                                pack,which_graph=comparison_type,
                                train_conf=train_conf,test_conf=test_conf,
                                global_deg_metric=global_deg_metric,local_deg_metric=local_deg_metric
                                )
                
                plot_times_best_by_deg([axs[i,0],axs[i,4],axs[i,8]],matrix1,pack_name=pack.replace('pack_PPO_',''),title=title[0],not_last_pack=not_last_pack)
                plot_times_best_by_deg([axs[i,1],axs[i,5],axs[i,9]],matrix2,title=title[1],not_last_pack=not_last_pack)
                plot_times_best_by_deg([axs[i,2],axs[i,6],axs[i,10]],matrix3,title=title[2],not_last_pack=not_last_pack)
                
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
            
            if not comparison_type=='how_times_best':
                for i in range(len(all_packs)):
                    axs[i,3].axis('off')
                    axs[i,7].axis('off')
            else:
                for i in range(3):
                    axs[p,i].axis('off')
                    axs[p*2+1,i].axis('off')

        # Guardar en df datos de interes 
        if comparison_type=='how_times_best':        
            df = pd.DataFrame(columns=['case', 'mean_perc_init', 'mean_perc_learn','mean_perc_conv','mean_perc_all'])
            
            df.loc[len(df)] =['last wins train']+list(np.array([last_train[0][0],last_train[1][0],last_train[2][0]])/len(self.all_packs))+[sum([last_train[0][0],last_train[1][0],last_train[2][0]])/(3*len(self.all_packs))]
            df.loc[len(df)] =['train wins last']+list(np.array([last_train[0][2],last_train[1][2],last_train[2][2]])/len(self.all_packs))+[sum([last_train[0][2],last_train[1][2],last_train[2][2]])/(3*len(self.all_packs))]
            df.loc[len(df)] =['last wins test']+list(np.array([last_test[0][0],last_test[1][0],last_test[2][0]])/len(self.all_packs))+[sum([last_test[0][0],last_test[1][0],last_test[2][0]])/(3*len(self.all_packs))]
            df.loc[len(df)] =['test wins last']+list(np.array([last_test[0][2],last_test[1][2],last_test[2][2]])/len(self.all_packs))+[sum([last_test[0][2],last_test[1][2],last_test[2][2]])/(3*len(self.all_packs))]
            df.loc[len(df)] =['train wins test']+list(np.array([train_test[0][0],train_test[1][0],train_test[2][0]])/len(self.all_packs))+[sum([train_test[0][0],train_test[1][0],train_test[2][0]])/(3*len(self.all_packs))]
            df.loc[len(df)] =['test wins train']+list(np.array([train_test[0][2],train_test[1][2],train_test[2][2]])/len(self.all_packs))+[sum([train_test[0][2],train_test[1][2],train_test[2][2]])/(3*len(self.all_packs))]
            df.loc[len(df)] =['last ties train']+list(np.array([last_train[0][1],last_train[1][1],last_train[2][1]])/len(self.all_packs))+[sum([last_train[0][1],last_train[1][1],last_train[2][1]])/(3*len(self.all_packs))]
            df.loc[len(df)] =['last ties test']+list(np.array([last_test[0][1],last_test[1][1],last_test[2][1]])/len(self.all_packs))+[sum([last_test[0][1],last_test[1][1],last_test[2][1]])/(3*len(self.all_packs))]
            df.loc[len(df)] =['train ties test']+list(np.array([train_test[0][1],train_test[1][1],train_test[2][1]])/len(self.all_packs))+[sum([train_test[0][1],train_test[1][1],train_test[2][1]])/(3*len(self.all_packs))]

            df.to_csv(self.data_path+'/paper/df_perc_times_best.csv', index=False)

        if comparison_type=='in_which_deg_best':
            top_header = [
                "Case",
                "Initialization","Initialization","Initialization",
                "Learning","Learning","Learning",
                "Convergence","Convergence","Convergence",
                "All regions","All regions","All regions"
            ]

            sub_header = [
                "",
                "Low", "Middle", "High",
                "Low", "Middle", "High",
                "Low", "Middle", "High",
                "Low", "Middle", "High",
            ]

            columns = pd.MultiIndex.from_arrays([top_header, sub_header])

            df = pd.DataFrame(columns=columns)

            def plain_list(list):
                return np.array([x for sublista in list for x in sublista])
            
            #df.loc[len(df)] =['last wins train']+list(plain_list(last_train[0])/(plain_list(last_train[0])+plain_list(last_train[1])))
            df.loc[len(df)] =['train wins last']+list(plain_list(last_train[1])/(plain_list(last_train[0])+plain_list(last_train[1])))
            #df.loc[len(df)] =['last wins test']+list(plain_list(last_test[0])/(plain_list(last_test[0])+plain_list(last_test[1])))
            df.loc[len(df)] =['test wins last']+list(plain_list(last_test[1])/(plain_list(last_test[0])+plain_list(last_test[1])))
            #df.loc[len(df)] =['train wins test']+list(plain_list(train_test[0])/(plain_list(train_test[0])+plain_list(train_test[1])))
            df.loc[len(df)] =['test wins train']+list(plain_list(train_test[1])/(plain_list(train_test[0])+plain_list(train_test[1])))

            df.to_csv(self.data_path + '/paper/df_conditioned_deg_perc.csv', index=False)

            df.to_latex(self.data_path + '/paper/df_conditioned_deg_perc.tex',
                index=False,
                escape=False,
                float_format="%.2f",
                multicolumn=True,
                multicolumn_format='c',
                column_format="|c|ccc|ccc|ccc|ccc|"
            )
   
        plt.savefig(self.figure_path+'/all_setups/criteria_comparison_'+comparison_type+'.pdf')
        
    def criteria_consequences(self,consequence_type='learning_curve'):

        df_conf=pd.read_csv(self.data_path+'/configurations.csv')
        conf_str=df_conf.loc[df_conf["pack"] == 'all', "test_cost_freq_opt"].iloc[0]
        test_cost_freq=conf_str.split('_')[0]+'_'+conf_str.split('_')[1]+'cost'

        legend_elements = [Line2D([0], [0], color='black', label='Truth best', linestyle='-', linewidth=1.5),
                            Line2D([0], [0], marker='o', color='blue', label='Last',markersize=6, linestyle=''),
                            Line2D([0], [0], marker='^', color='orange', label='Default train',markersize=6, linestyle=''),
                            Line2D([0], [0], marker='D', color="#A52D81", label='Default validation',markersize=6, linestyle=''),
                            Line2D([0], [0], marker='s', color='green', label='Cost-driven validation',markersize=6, linestyle='')
                            
                        ]


        def plot_pack_criteria_learning_curves(axs,pack_name,pack_indx,first_pack=False):

            def plot_mediana_ci(ax, df, color, marker=None):
                ax.plot(df.index , df.median(axis=1), color=color, marker=marker,markevery=int(len(df.index)/10),linewidth=1)
                ax.fill_between(df.index, df.quantile(0.25, axis=1), df.quantile(0.75, axis=1), color=color, alpha=0.2)
            

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
            
        def plot_pack_criteria_cummulative_diff(axs,pack_name,pack_indx,first_pack=False,diff_type='cummulative'):
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
            def plot_cummulative_learning_curve_paired_diff(ax,data,color):
                data = np.array(data)
                accumulated = np.zeros(data.shape[1])
                
                for seed in data:
                    new_accumulated = accumulated + seed
                    ax.fill_between(np.arange(data.shape[1]),accumulated,new_accumulated,color=color,alpha=0.6,edgecolor='none') # area entre la curva anterior y la nueva
                    ax.plot(np.arange(data.shape[1]),new_accumulated,color=color,linewidth=0.8) # Curva superior de esta capa
                    accumulated = new_accumulated
            
            def plot_cummulative_mean_learning_curves(ax,data,color,marker):
                ax.plot(range(len(data)),data,color=color,marker=marker,markevery=int(len(data)*0.2))

            if diff_type=='cummulative':

                plot_cummulative_learning_curve_paired_diff(axs[0,pack_indx],last_paired_diff,'blue')
                plot_cummulative_learning_curve_paired_diff(axs[1,pack_indx],train_paired_diff,'orange')
                plot_cummulative_learning_curve_paired_diff(axs[2,pack_indx],test_paired_diff,'#A52D81')
                plot_cummulative_learning_curve_paired_diff(axs[3,pack_indx],cost_freq_paired_diff,"green")

                if first_pack:
                    if pack_name!='':
                        axs[3,pack_indx].legend(handles=legend_elements,loc='upper center',bbox_to_anchor=(1.5, -0.3),ncol=5,frameon=False)
                    for i in range(4):
                        axs[i,pack_indx].set_ylabel(r"$\sum_{\rho}~f(\pi^*_t)-f(\widetilde{\pi}_t)$",fontsize=10)
                for i in range(4):
                    axs[i, pack_indx].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
                    if i!=0:
                        axs[i, pack_indx].yaxis.get_offset_text().set_visible(False)

                axs[0,pack_indx].set_title(abbreviate(pack_name),fontsize=12)
                axs[3,pack_indx].set_xlabel(r"$t$",fontsize=12)

            if diff_type=='mean':
                plot_cummulative_mean_learning_curves(axs[pack_indx],[0]*len(last_paired_diff[-1]),'black',None)
                plot_cummulative_mean_learning_curves(axs[pack_indx],[sum(col) / len(col) for col in zip(*last_paired_diff)],'blue','o')
                plot_cummulative_mean_learning_curves(axs[pack_indx],[sum(col) / len(col) for col in zip(*train_paired_diff)],'orange','^')
                plot_cummulative_mean_learning_curves(axs[pack_indx],[sum(col) / len(col) for col in zip(*test_paired_diff)],'#A52D81','D')
                plot_cummulative_mean_learning_curves(axs[pack_indx],[sum(col) / len(col) for col in zip(*cost_freq_paired_diff)],"green",'s')
                
                if first_pack:
                    if pack_name!='':
                        axs[pack_indx].legend(handles=legend_elements,loc='upper center',bbox_to_anchor=(2, -0.2),ncol=5,frameon=False,fontsize=12)
                        axs[pack_indx].set_ylabel(r"$\text{mean}_{\rho}~f(\pi^*_t)-f(\widetilde{\pi}_t)$",fontsize=14)

                axs[pack_indx].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
                axs[pack_indx].set_title(abbreviate(pack_name),fontsize=14)
                axs[pack_indx].set_xlabel(r"$t$",fontsize=14)
                axs[pack_indx].tick_params(axis='both', labelsize=10)

        def plot_pack_early_stopping(axs,pack_name,pack_indx,first_pack=False,early_stopping_graph='all'):

            test_n_ep_default,test_freq_default=int(test_default.split('_')[0]),int(test_default.split('_')[1])
            test_n_ep,test_freq=int(conf_str.split('_')[0]),float(conf_str.split('_')[1])

            # Obtener datos para las graficas
            matrix_best_est,matrix_best_truth=EarlyStopping.successive_halving(self.data_path+'/',pack)
            matrix_last_est,matrix_last_truth=EarlyStopping.successive_halving(self.data_path+'/',pack,criteria='last')
            matrix_train_default_est,matrix_train_default_truth=EarlyStopping.successive_halving(self.data_path+'/',pack,criteria='train',conf=[train_default,None])
            matrix_test_default_est,matrix_test_default_truth=EarlyStopping.successive_halving(self.data_path+'/',pack,criteria='test',conf=[test_n_ep_default,test_freq_default])
            matrix_test_est,matrix_test_truth=EarlyStopping.successive_halving(self.data_path+'/',pack,criteria='test',conf=[test_n_ep,test_freq])

            
            # Grafica 1: sucessive halving explicito
            def plot_succesive_halving(ax,matrix_criteria,color):
                max_long=max([len(i) for i in matrix_criteria])
                for truth_evol in matrix_criteria:
                    ax.plot(range(len(truth_evol)) , truth_evol, color=color,linewidth=1)  
                    if len(truth_evol)!=max_long:
                        ax.axvline(x=len(truth_evol), color='red', linewidth=1)

            if early_stopping_graph=='all':
                plot_succesive_halving(axs[0,pack_indx],matrix_best_truth,'black')
                plot_succesive_halving(axs[1,pack_indx],matrix_last_truth,'blue')
                plot_succesive_halving(axs[2,pack_indx],matrix_train_default_truth,'orange')
                plot_succesive_halving(axs[3,pack_indx],matrix_test_default_truth,'purple')
                plot_succesive_halving(axs[4,pack_indx],matrix_test_truth,'green')

                if first_pack:
                    for i,ylabel in enumerate(['Ground truth\n','Last\n','Default train\n','Default validation\n','Cost-driven\nvalidation']):
                        axs[i,pack_indx].set_ylabel(ylabel+'\n'+r"$f(\widetilde{\pi}_t)$",fontsize=12)
                for i in range(5):
                    axs[i, pack_indx].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
                    if i!=0:
                        axs[i, pack_indx].yaxis.get_offset_text().set_visible(False)


                axs[0,pack_indx].set_title(abbreviate(pack_name),fontsize=12)
                axs[4,pack_indx].set_xlabel(r"$t$",fontsize=12)

            # Graficas 2 y 3: info de resource allocation (eff) y evolucion de performnace (prec)
            def plot_resource_allocation(ax, listas, colors):

                block_centers = []

                # markers por color (orden fijo)
                marker_map = {'blue': 'o','orange': '^','green': 's','purple': 'D'}

                # 1. rango global
                all_data = [x for lista in listas for sub in lista for x in sub]
                min_val = min(all_data)
                max_val = max(all_data)

                # bins no uniformes
                bins = np.array([min_val,min_val + 0.25 * (max_val - min_val),min_val + 0.75 * (max_val - min_val),max_val])

                n_groups = len(listas)
                bar_height = (max_val - min_val) * 0.06
                spacing = bar_height * 1.15
                outer_margin = (max_val - min_val) * 0.08

                # 2. histogramas
                histogramas = []
                for lista_de_listas in listas:
                    datos = [x for sub in lista_de_listas for x in sub]
                    counts, _ = np.histogram(datos, bins=bins)
                    perc = counts / len(datos)
                    histogramas.append(perc)

                # 3. dibujar
                for b in range(len(bins) - 1):

                    y_center = (bins[b] + bins[b+1]) / 2
                    block_centers.append(y_center)

                    total_height = (n_groups - 1) * spacing
                    start = y_center - total_height / 2

                    for i, counts in enumerate(histogramas):
                        y = start + i * spacing
                        ax.barh(y,counts[b],height=bar_height,left=0,color=colors[i],align='center',alpha=0.9)

                        color = colors[i]
                        if color in marker_map and color != 'black':
                            ax.scatter(counts[b],y,marker=marker_map[color],color=colors[i],edgecolor='black',zorder=5,s=30)

                # líneas separadoras entre bloques
                for i in range(len(block_centers) - 1):

                    y_sep = (block_centers[i] + block_centers[i+1]) / 2

                    ax.axhline(
                        y=y_sep,
                        color='black',
                        linewidth=0.8,
                        zorder=10
                    )

                # 4. estetica
                ax.set_ylim(min_val - outer_margin, max_val + outer_margin)  
                ax.set_xlim(0, max(max(h) for h in histogramas) * 1.1)

                bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
                ax.set_yticks(bin_centers)

                ax.set_xlabel('Proportion of evaluations')

                # para que los numeros del eje x no se superpongan
                step = 100

                xmin, xmax = ax.get_xlim()

                tick_1 = step * round((xmin + (xmax - xmin)/3) / step,3)
                tick_2 = 2 * tick_1

                ax.set_xticks([0, tick_1, tick_2])
                ax.set_xticklabels([0, tick_1, tick_2])
                        
                if pack_indx == 0:
                    ax.set_ylabel('Truth reward values')
                    ax.set_yticklabels(["low", "middle", "high"])
                    ax.legend(handles=legend_elements,loc='upper center',bbox_to_anchor=(2, -0.25),ncol=5,frameon=False)
                else:
                    ax.set_yticklabels(["", "", ""])    

            # def plot_resource_allocation(ax, listas, colors):

            #     mpl.rcParams['hatch.linewidth'] = 0.6
            #     block_centers = []

            #     # textura por color
            #     hatch_map = {
            #         'blue': '..',
            #         'orange': '//',
            #         'green': '||',
            #         'purple': 'xx',
            #         'black': None
            #     }

            #     # 1. rango global
            #     all_data = [x for lista in listas for sub in lista for x in sub]
            #     min_val = min(all_data)
            #     max_val = max(all_data)

            #     bins = np.array([
            #         min_val,
            #         min_val + 0.25 * (max_val - min_val),
            #         min_val + 0.75 * (max_val - min_val),
            #         max_val
            #     ])

            #     n_groups = len(listas)
            #     bar_height = (max_val - min_val) * 0.06
            #     spacing = bar_height * 1.2
            #     outer_margin = (max_val - min_val) * 0.08

            #     # 2. histogramas
            #     histogramas = []
            #     for lista_de_listas in listas:
            #         datos = [x for sub in lista_de_listas for x in sub]
            #         counts, _ = np.histogram(datos, bins=bins)
            #         histogramas.append(counts)

            #     # 3. dibujar
            #     for b in range(len(bins) - 1):

            #         y_center = (bins[b] + bins[b+1]) / 2
            #         block_centers.append(y_center)

            #         total_height = (n_groups - 1) * spacing
            #         start = y_center - total_height / 2

            #         for i, counts in enumerate(histogramas):

            #             y = start + i * spacing

            #             ax.barh(
            #                 y,
            #                 counts[b],
            #                 height=bar_height,
            #                 left=0,
            #                 color=colors[i],
            #                 hatch=hatch_map.get(colors[i], None),
            #                 edgecolor='black' if colors[i] != 'black' else 'black',
            #                 linewidth=0
            #             )

            #     # líneas separadoras entre bloques
            #     for i in range(len(block_centers) - 1):

            #         y_sep = (block_centers[i] + block_centers[i+1]) / 2

            #         ax.axhline(
            #             y=y_sep,
            #             color='black',
            #             linewidth=0.8,
            #             zorder=10
            #         )

            #     # 4. estética
            #     ax.set_ylim(min_val - outer_margin, max_val + outer_margin)
            #     ax.set_xlim(0, max(max(h) for h in histogramas) * 1.1)

            #     bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
            #     ax.set_yticks(bin_centers)

            #     ax.set_xlabel('Number of evaluations')

            #     step = 100
            #     xmin, xmax = ax.get_xlim()

            #     tick_1 = step * round((xmin + (xmax - xmin)/3) / step)
            #     tick_2 = 2 * tick_1

            #     ax.set_xticks([0, tick_1, tick_2])

            #     if pack_indx == 0:
            #         ax.set_ylabel('Truth values')
            #         ax.set_yticklabels(["low", "middle", "high"])


            #         legend_elements = [
            #             mpatches.Patch(
            #                 facecolor='black',
            #                 edgecolor='black',
            #                 label='truth best'
            #             ),
            #             mpatches.Patch(
            #                 facecolor='blue',
            #                 edgecolor='black',
            #                 hatch='..',
            #                 label='last'
            #             ),
            #             mpatches.Patch(
            #                 facecolor='orange',
            #                 edgecolor='black',
            #                 hatch='//',
            #                 label='train'
            #             ),
            #             mpatches.Patch(
            #                 facecolor='purple',   # granate aproximado
            #                 edgecolor='black',
            #                 hatch='xx',
            #                 label='default test'
            #             ),
            #             mpatches.Patch(
            #                 facecolor='green',
            #                 edgecolor='black',
            #                 hatch='||',
            #                 label='cost-driven test'
            #             )
            #         ]

            #         ax.legend(
            #             handles=legend_elements,
            #             loc='upper center',
            #             bbox_to_anchor=(1.8, -0.25),
            #             ncol=5,
            #             frameon=False,
            #             handleheight=1.2
            #         )
            #     else:
            #         ax.set_yticklabels(["", "", ""])

            def accuracy_evol(matrix_criteria_est,matrix_criteria_truth,plot_yes=False,
                                       curve_truth=None,curve_criteria=None,current_min=None):

                    
                if plot_yes:
                    acc_evol=[]
                    for i in range(len(curve_truth)):
                        numerator=curve_criteria[i]-current_min[i]
                        denominator=curve_truth[i]-current_min[i]
                        
                        acc_evol.append([1 if denominator==0 or curve_criteria[i]>curve_truth[i] else numerator/denominator])

                    return acc_evol
                else:

                    max_long=max([len(i) for i in matrix_criteria_truth])

                    current_best=[]
                    for i in range(max_long):
                        best_by_process_est=[]
                        best_by_process_truth=[]
                        for sub_est,sub_truth in zip(matrix_criteria_est,matrix_criteria_truth):
                            if len(sub_truth)>=i+1:
                                best_by_process_est.append(sub_est[i])
                                best_by_process_truth.append(sub_truth[i])
                        current_best.append(best_by_process_truth[best_by_process_est.index(max(best_by_process_est))]) 

                    return current_best
                
            def get_current_min(matrix_truth,matrix_criteria):

                max_long=max([len(i) for i in matrix_truth])
                current_min=[]
                for i in range(max_long):
                    # Minimo global para normalizar despues
                    current_truht_mins,current_criteria_mins=[],[]
                    for sub_truth,sub_criteria in zip(matrix_truth,matrix_criteria):
                        current_truht_mins.append(min(sub_truth[:i+1]))
                        current_criteria_mins.append(min(sub_criteria[:i+1]))
                    current_min.append(min(current_truht_mins+current_criteria_mins))

                return current_min

            def plot_pack_acc_CIs(ax,acc_lists,pack,first_pack=False,xlabel=False):

                colors=['blue','orange',"#A52D81",'green']
                markers=['o','^','D','s']

                for i, data in enumerate(acc_lists):

                    p25 = np.percentile(data, 25)
                    med = np.median(data)
                    p75 = np.percentile(data, 75)

                    ax.hlines(i, p25, p75, color=colors[i], linewidth=1)
                    ax.vlines([p25, p75], i-0.15, i+0.15, color=colors[i], linewidth=1)
                    ax.plot(med, i, markers[i], color=colors[i], markersize=6)

                    ax.set_title(abbreviate(pack.replace('pack_PPO_','')))
                    ax.set_xlim(-0.1,1.1)
                    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
                    ax.set_xticklabels([0, 0.25, 0.5, 0.75, 1])
                    ax.grid(axis='x', linestyle='--', linewidth=0.8, alpha=0.6)
                    if first_pack:
                        ax.set_yticks(range(len(acc_lists)))
                        ax.set_yticklabels(['Last', 'Default train', 'Default\nvalidation', 'Cost-driven\nvalidation'])
                    else:
                        ax.set_yticklabels([]) 
                        ax.tick_params(axis='y', which='both', left=True, right=False, length=5)

                    if xlabel:
                        ax.set_xlabel(r'$\alpha^{SH}$')

            if early_stopping_graph=='eff':
                plot_resource_allocation(axs[pack_indx],
                                 [matrix_best_truth,matrix_last_truth,matrix_train_default_truth,matrix_test_default_truth,matrix_test_truth],
                                 ['black','blue','orange','purple','green'])
                axs[pack_indx].set_title(abbreviate(pack_name))

            if early_stopping_graph=='prec':
  
                truth_curve=accuracy_evol(matrix_best_est,matrix_best_truth)
                last_curve=accuracy_evol( matrix_last_est,matrix_last_truth)
                traind_curve=accuracy_evol( matrix_train_default_est,matrix_train_default_truth)
                testd_curve=accuracy_evol( matrix_test_default_est,matrix_test_default_truth)
                test_curve=accuracy_evol( matrix_test_est,matrix_test_truth)

                current_min=get_current_min(matrix_best_truth,matrix_best_truth)
                accuracy_evol( matrix_best_est,matrix_best_truth,plot_yes=True,curve_truth=truth_curve,curve_criteria=truth_curve,current_min=current_min)
                current_min=get_current_min(matrix_best_truth,matrix_last_truth)
                acc_last=accuracy_evol( matrix_last_est,matrix_last_truth,plot_yes=True,curve_truth=truth_curve,curve_criteria=last_curve,current_min=current_min)
                current_min=get_current_min(matrix_best_truth,matrix_train_default_truth)
                acc_traind=accuracy_evol( matrix_train_default_est,matrix_train_default_truth,plot_yes=True,curve_truth=truth_curve,curve_criteria=traind_curve,current_min=current_min)
                current_min=get_current_min(matrix_best_truth,matrix_test_default_truth)
                acc_testd=accuracy_evol( matrix_test_default_est,matrix_test_default_truth,plot_yes=True,curve_truth=truth_curve,curve_criteria=testd_curve,current_min=current_min)
                current_min=get_current_min(matrix_best_truth,matrix_test_truth)
                acc_test=accuracy_evol( matrix_test_est,matrix_test_truth,plot_yes=True,curve_truth=truth_curve,curve_criteria=test_curve,current_min=current_min)

                first_pack=[True if pack_indx==0 else False][0]
                xlabel=[True if pack_indx==len(all_packs)//2 else False][0]
                plot_pack_acc_CIs(axs[pack_indx],[acc_last,acc_traind,acc_testd,acc_test],pack,first_pack=first_pack,xlabel=xlabel)

                


        if consequence_type in ['learning_curve','cummulative_diff']:
            fig, axs = plt.subplots(4,len(self.all_packs), figsize=(3*len(self.all_packs),6),sharex='col',sharey='col')
            plt.subplots_adjust(top=0.95,bottom=0.1,left=0.05,right=0.98, hspace=0.05,wspace=0.15)
        if consequence_type=='early_stopping_all':
            fig, axs = plt.subplots(5,len(self.all_packs), figsize=(2.5*len(self.all_packs),9),sharex='col',sharey='col')
            plt.subplots_adjust(top=0.95,bottom=0.05,left=0.06,right=0.98, hspace=0.05,wspace=0.25)
        if consequence_type =='early_stopping_eff':
            fig, axs = plt.subplots(1,len(self.all_packs), figsize=(2*len(self.all_packs),3))
            plt.subplots_adjust(top=0.9,bottom=0.25,left=0.07,right=0.98, hspace=0.0,wspace=0.05)
        if consequence_type in['cummulative_mean_diff','early_stopping_prec']:
            fig, axs = plt.subplots(1,len(self.all_packs), figsize=(2*len(self.all_packs),2))
            plt.subplots_adjust(top=0.85,bottom=0.2,left=0.08,right=0.99, hspace=0.0,wspace=0.05)

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
            if consequence_type=='cummulative_mean_diff':
                plot_pack_criteria_cummulative_diff(axs,pack.replace('pack_PPO_',''),i,first_pack=first_pack,diff_type='mean')
            if consequence_type=='early_stopping_all':
                plot_pack_early_stopping(axs,pack.replace('pack_PPO_',''),i,first_pack=first_pack)
            if consequence_type =='early_stopping_eff':
                plot_pack_early_stopping(axs,pack.replace('pack_PPO_',''),i,first_pack=first_pack,early_stopping_graph='eff')
            if consequence_type =='early_stopping_prec':
                plot_pack_early_stopping(axs,pack.replace('pack_PPO_',''),i,first_pack=first_pack,early_stopping_graph='prec')
            
        plt.savefig(self.figure_path+'/all_setups/criteria_consequences_'+consequence_type+'.pdf')

    def test_default_vs_with_cost(self):

        # Configuracion de test with cost
        df_conf=pd.read_csv(self.data_path+'/configurations.csv')
        conf_str=df_conf.loc[df_conf["pack"] == 'all', "test_cost_freq_opt"].iloc[0]
        test_conf=conf_str.split('_')[0]+'_'+conf_str.split('_')[1]+'cost'

        # Funciones
        def plot_pack_costs_accs(ax,cost_list,pack_name='',ytitle='',title='',not_last_pack=True,first_pack=False,first_column=True):
 
            # Nube de puntos
            sns.stripplot(x=cost_list, orient='h', jitter=0.4,ax=ax,color='black',zorder=1,size=1)

            # Linea de la mediana
            ax.axvline(np.median(cost_list), color='red',linewidth=2,zorder=2)
            ax.axvspan(np.quantile(cost_list,0.25), np.quantile(cost_list,0.75), color='red', alpha=0.5,zorder=2)

            ax.set_xlim(-0.05, 1.05)
            ax.text(-0.4, -2.5, ytitle,transform=ax.transAxes,rotation=90,va="center",ha="center",fontsize=10)
            if first_column:
                ax.set_ylabel(abbreviate(pack_name), rotation=0,fontsize=10)
                ax.yaxis.set_label_coords(-0.15, 0.1) 

            ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_xticklabels(['0', '0.25', '0.5', '0.75', '1'],fontsize=10)

            if not_last_pack:   
                ax.tick_params(axis='x', labelbottom=False)
            if first_pack:
                ax.set_title(title,fontsize=10)
            ax.set_yticks([]) 

        def plot_val_times(ax,times_list,single_or_list='list',pack_name='',not_last_pack=True,first_column=True,first_pack=False):

            if single_or_list=='list':
                ax.barh(0, np.median(times_list), height=0.3, color='black')
                ax.axvspan(np.quantile(times_list,0.25), np.quantile(times_list,0.75), color='red', alpha=0.5,zorder=2)

            if single_or_list=='number':
                ax.barh(0, times_list, height=0.3, color='black')
            
            ax.set_xlim(-0.05, 1.05)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_xticklabels(['0', '0.25', '0.5', '0.75', '1'],fontsize=10)
            
            if first_column:
                ax.set_ylabel(abbreviate(pack_name), rotation=0,fontsize=10)
                ax.yaxis.set_label_coords(-0.15, 0.1) 
            if not_last_pack:   
                ax.tick_params(axis='x', labelbottom=False)
            if first_pack:
                ax.set_title('Proportion of validations\n',fontsize=10)
            ax.set_yticks([]) 
                
        # Grafica
        #plt.rcParams.update({"font.family": "cmr10","mathtext.fontset": "cm", })
        fig, axs = plt.subplots(len(self.all_packs)*2+1,5, figsize=(3*2,0.4*len(self.all_packs)),
                                    gridspec_kw={'width_ratios': [1,0,1,0,1],
                                                 'height_ratios': [1]*len(self.all_packs)+[0.5]+[1]*len(self.all_packs)})
        plt.subplots_adjust(top=0.87,bottom=0.1,left=0.15,right=0.98, hspace=0.0,wspace=0.0)

        axs = np.atleast_2d(axs)

        for i,pack in enumerate(all_packs):

            
            # Leear bases de datos necesarias
            df_cost=pd.read_csv(self.data_path+'/'+pack.replace('pack',self.library)+'/df_test_cost.csv')
            df_n_ep=pd.read_csv(self.data_path+'/'+pack.replace('pack',self.library)+'/df_test_n_ep.csv')
            df_acc=pd.read_csv(self.data_path+'/'+pack.replace('pack',self.library)+'/df_test_prec.csv')

            # Listas de costes
            test_default=df_conf.loc[df_conf["pack"] == pack, "test_default"].iloc[0]
            df_default= df_cost.loc[:, df_cost.columns.str.contains('_'+test_default+'_')]
            cost_default = [min(cost, 1) for cost in df_default.values.flatten().tolist()] 
            df_with_cost=df_cost.loc[:, df_cost.columns.str.contains('_'+test_conf+'_')]
            cost_with_cost=[min(cost, 1) for cost in df_with_cost.values.flatten().tolist()] 

            # Listas de numero de validaciones
            times_default=int(df_n_ep.shape[0]/int(test_default.split('_')[1]))/df_n_ep.shape[0]
            df_with_cost=df_n_ep.loc[:, df_n_ep.columns.str.contains('_'+test_conf+'_')]
            times_with_cost=[i/df_n_ep.shape[0] for i in (df_with_cost != 0).sum().tolist()]

            # Listas de accuracies
            df_default= df_acc.loc[:, df_acc.columns.str.contains('_'+test_default+'_')]
            acc_default=[acc for acc in df_default.values.flatten().tolist()] 
            df_default= df_acc.loc[:, df_acc.columns.str.contains('_'+test_conf+'_')]
            acc_with_cost=[acc for acc in df_default.values.flatten().tolist()] 


            not_last_pack=[False if i==len(all_packs)-1 else True][0]
            first_pack=[True if i==0 else False][0]
            titles=[['Default\nvalidation', 'Cost-driven\nvalidation']if i==0 else ['','']][0]
            plot_pack_costs_accs(axs[i,0],cost_default,pack_name=pack.replace('pack_PPO_',''),ytitle=titles[0],not_last_pack=True,first_pack=first_pack,title='Validation-to-learning\ncost ratio')
            plot_pack_costs_accs(axs[i+len(self.all_packs)+1,0],cost_with_cost,pack_name=pack.replace('pack_PPO_',''),ytitle=titles[1],not_last_pack=not_last_pack)
            plot_val_times(axs[i,2],times_default,pack_name=pack.replace('pack_PPO_',''),single_or_list='number',not_last_pack=True,first_column=False,first_pack=first_pack)
            plot_val_times(axs[i+len(self.all_packs)+1,2],times_with_cost,pack_name=pack.replace('pack_PPO_',''),single_or_list='list',not_last_pack=not_last_pack,first_column=False)
            plot_pack_costs_accs(axs[i,4],acc_default,pack_name=pack.replace('pack_PPO_',''),not_last_pack=True,first_column=False,first_pack=first_pack,title='Accuracies\n')
            plot_pack_costs_accs(axs[i+len(self.all_packs)+1,4],acc_with_cost,pack_name=pack.replace('pack_PPO_',''),not_last_pack=not_last_pack,first_column=False)

        for i in range(len(self.all_packs)):
            axs[i,1].axis('off')
            axs[i,3].axis('off')
            axs[len(self.all_packs)+i+1,1].axis('off')
            axs[len(self.all_packs)+i+1,3].axis('off')

        for i in range(5):
            axs[len(self.all_packs),i].axis('off')


        plt.savefig('experiments/results/figures/all_setups/test_default_vs_with_cost.png')

    def patterns_deg_acc_eff(self):

        # Plotear scatter plot para pack-region
        def plot_uncertainty_rectangles(ax, lista1, lista2, lista3, lista4,color,pack='',first_row=True,first_column=True,center_row=True):

            x_med = np.median(lista1)
            y_lists = [lista2, lista3, lista4]
            markers = ['o', '^', 's'] 

            for y_data, marker in zip(y_lists, markers):
                y_med = np.median(y_data)
                ax.scatter(x_med,y_med,facecolors='none',edgecolor=color,marker=marker,s=40,zorder=3)

                ax.set_ylim(-0.15,1.15)
                ax.set_xlim(-0.1,1.1)
                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])

                ax.axvline(0.25, color='grey', linestyle='-', alpha=0.1,linewidth=0.5)
                ax.axvline(0.75, color='grey', linestyle='-', alpha=0.1,linewidth=0.5)
                ax.axhline(0.25, color='grey', linestyle='-', alpha=0.1,linewidth=0.5)
                ax.axhline(0.75, color='grey', linestyle='-', alpha=0.1,linewidth=0.5)

                
                if first_row:
                    ax.set_title(abbreviate(pack.replace('pack_PPO_','')),fontsize=9)
                    ax.set_xticks([])
                else:
                    if center_row: 
                        ax.set_xlabel(r'$\delta$')
                if not first_column:
                    ax.set_yticks([])
                else:
                    if first_row:
                        ax.set_ylabel(r'$\alpha$')
                    else:
                        ax.set_ylabel(r'$\varepsilon$')

        
        # Cargar datos necesarios
        df_conf=pd.read_csv(self.data_path+'/configurations.csv')
        train_conf=str(int(df_conf.loc[df_conf["pack"] == 'all', "train_opt"].iloc[0]))
        conf_str=df_conf.loc[df_conf["pack"] == 'all', "test_cost_freq_opt"].iloc[0]
        test_conf=conf_str.split('_')[0]+'_'+conf_str.split('_')[1]+'cost'
            
        # Graficas
        fig, axs = plt.subplots(2,len(self.all_packs), figsize=(1.3*len(self.all_packs),1.2*2))
        plt.subplots_adjust(top=0.9,bottom=0.16,left=0.05,right=0.8, hspace=0,wspace=0)

        
        for i,pack in enumerate(all_packs):
            first_column=[True if i==0 else False][0]
            center_row=[True if i==len(all_packs)//2 else False][0]
            colors = ["#693106", "#BB8A20", "#bba86a"][::-1]
            pack_path=self.data_path+'/'+pack.replace('pack',self.library)+'/'

            #---- Degradacion
            deg1,deg2,deg3=DataConverter.from_df_data_to_graph_data(
                                    [pack_path+'learning_regions.csv',pack_path+'deg_evolution.csv'],pack,'deg_distribution',
                                    global_deg_metric=global_deg_metric,local_deg_metric=local_deg_metric
                                ) 

            #---- Precision
            last1,last2,last3=DataConverter.from_df_data_to_graph_data(
                            [pack_path+'learning_regions.csv',pack_path+'df_last_prec.csv'],pack,which_graph='last_prec',
                            prec_metric=prec_metric,eff_metric=eff_metric
                        )
            train1,train2,train3=DataConverter.from_df_data_to_graph_data(
                            [pack_path+'learning_regions.csv',pack_path+'df_train_prec.csv'],pack,which_graph='train_prec',
                            prec_metric=prec_metric,eff_metric=eff_metric,train_conf=train_conf
                        )
            test1,test2,test3=DataConverter.from_df_data_to_graph_data(
                            [pack_path+'learning_regions.csv',pack_path+'df_test_prec.csv'],pack,which_graph='test_prec',
                            prec_metric=prec_metric,eff_metric=eff_metric,test_conf=test_conf
                        )
            
            plot_uncertainty_rectangles(axs[0,i],deg1,last1,train1,test1,colors[0],pack,first_column=first_column,center_row=center_row)
            plot_uncertainty_rectangles(axs[0,i],deg2,last2,train2,test2,colors[1],pack,first_column=first_column,center_row=center_row)
            plot_uncertainty_rectangles(axs[0,i],deg3,last3,train3,test3,colors[2],pack,first_column=first_column,center_row=center_row)
            
            #---- Eficacia
            last1,last2,last3=DataConverter.from_df_data_to_graph_data(
                            [pack_path+'learning_regions.csv',pack_path+'df_last_eff.csv'],pack,which_graph='last_eff',
                            prec_metric=prec_metric,eff_metric=eff_metric
                        )
            train1,train2,train3=DataConverter.from_df_data_to_graph_data(
                            [pack_path+'learning_regions.csv',pack_path+'df_train_eff.csv'],pack,which_graph='train_eff',
                            prec_metric=prec_metric,eff_metric=eff_metric,train_conf=train_conf
                        )
            test1,test2,test3=DataConverter.from_df_data_to_graph_data(
                            [pack_path+'learning_regions.csv',pack_path+'df_test_eff.csv'],pack,which_graph='test_eff',
                            prec_metric=prec_metric,eff_metric=eff_metric,test_conf=test_conf
                        )
            
            plot_uncertainty_rectangles(axs[1,i],deg1,last1,train1,test1,colors[0],pack,False,first_column=first_column,center_row=center_row)
            plot_uncertainty_rectangles(axs[1,i],deg2,last2,train2,test2,colors[1],first_row=False,first_column=first_column,center_row=center_row)
            plot_uncertainty_rectangles(axs[1,i],deg3,last3,train3,test3,colors[2],first_row=False,first_column=first_column,center_row=center_row)

        # Leyenda

        legend_elements = [
            # puntos (solo borde)
            Line2D([0], [0], marker='o', linestyle='None',
                markerfacecolor='none', markeredgecolor='black',
                label='Last'),

            Line2D([0], [0], marker='^', linestyle='None',
                markerfacecolor='none', markeredgecolor='black',
                label='Train'),

            Line2D([0], [0], marker='s', linestyle='None',
                markerfacecolor='none', markeredgecolor='black',
                label='Validation'),

            # líneas
            Line2D([0], [0], color=colors[0], lw=2,
                label='Initialization'),

            Line2D([0], [0], color=colors[1], lw=2,
                label='Learning'),

            Line2D([0], [0], color=colors[2], lw=2,
                label='Convergence'),
        ]

        axs[0,6].legend(
            handles=legend_elements,
            loc='upper center',
            bbox_to_anchor=(2, 1),  # fuera abajo
            ncol=1,                       # 3 columnas
            frameon=False,
            fontsize=10,
            handlelength=2
        )

            
        plt.savefig(self.figure_path+'/all_setups/patterns_degradation_precision_efficiency.pdf')
 

# Main program
grapher=Grapher(library,all_packs,seeds,data_path,figure_path) 

grapher.deg_prec_eff()
grapher.patterns_deg_acc_eff()
grapher.criteria_comparison()
grapher.criteria_comparison(comparison_type='in_which_deg_best')
grapher.criteria_comparison(comparison_type='how_times_best_in_deg_level')
#grapher.criteria_comparison(comparison_type='with_what_prec_diff_best')
#grapher.criteria_consequences()
#grapher.criteria_consequences(consequence_type='cummulative_diff')
#grapher.criteria_consequences(consequence_type='cummulative_mean_diff')
grapher.criteria_consequences(consequence_type='early_stopping_all')
grapher.criteria_consequences(consequence_type='early_stopping_eff')
grapher.criteria_consequences(consequence_type='early_stopping_prec')
grapher.test_default_vs_with_cost()