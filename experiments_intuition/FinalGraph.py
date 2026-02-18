from Main import *
from matplotlib.lines import Line2D


def from_df_data_to_graph_data(path,pack,
                                  which_graph=None,cost_prec=None,
                                  global_deg_metric=None,local_deg_metric=None,
                                  prec_metric=None,limit_metric=None):
    
    '''
    Extrae la informacion pertinente de las bases de datos de degradacion, precision o coste por regiones de aprendizaje, 
    en el formato apropiado para generar a partir de esos datos las graficas de interes.
    '''
    
    df_limits=pd.read_csv(path[0])
    df_limits = df_limits[
                    df_limits['pack_seed'].str.contains(pack, na=False) &
                    df_limits['limit_metric'].str.contains(limit_metric, na=False)
                ] # Solo filas del pack con la metrica de los limites indicada

    if which_graph=='deg_distribution':

        df_deg=pd.read_csv(path[1])
        df_deg = df_deg.filter(like=pack) # Solo columnas del pack
        df_deg = df_deg.filter(regex=global_deg_metric+'_'+local_deg_metric+"$") # Solo columnas del pack con deg indicada

        # Almacenar degradaciones por region
        deg_initialization,deg_learning,deg_stabilization=[],[],[]

        for pack_seed in tqdm(df_limits['pack_seed'],desc="Data for deg graphs"):
            a=int(df_limits.loc[df_limits['pack_seed'] == pack_seed, 'a'])
            b=int(df_limits.loc[df_limits['pack_seed'] == pack_seed, 'b'])

            df_deg_pack_seed= df_deg.filter(like=pack_seed+'_').iloc[:, 0].tolist()
            deg_initialization+=df_deg_pack_seed[:a]
            deg_learning+=df_deg_pack_seed[a:b]
            deg_stabilization+=df_deg_pack_seed[b:]

        return deg_initialization,deg_learning,deg_stabilization

    if which_graph=='train_conf_prec':

        def get_conf_from_column_name(column_names):
            conf_list=[]
            for column_name in column_names:
                splited=column_name.split('_')
                conf_list.append(splited[3])

            return sorted(list(set(conf_list)), key=int)[::-1]

        df_prec=pd.read_csv(path[1])
        df_prec = df_prec.filter(like=pack) # Solo columnas del pack
        df_prec = df_prec.filter(regex=prec_metric+"$") # Solo columnas del pack con metrica de prec indicada

        # Almacenar datos por configuracion para cada region
        matrix_conf_initialization, matrix_conf_learning,matrix_conf_stabilization=[],[],[]

        conf_list=get_conf_from_column_name(df_prec.columns)

        for conf in tqdm(conf_list,desc="Data for train_prec graphs"):
            prec_initialization,prec_learning,prec_stabilization=[],[],[]

            df_prec_conf=df_prec.filter(like='_'+conf+'_')

            for pack_seed in df_limits['pack_seed']:
                a=int(df_limits.loc[df_limits['pack_seed'] == pack_seed, 'a'])
                b=int(df_limits.loc[df_limits['pack_seed'] == pack_seed, 'b'])

                df_prec_conf_pack_seed= df_prec_conf.filter(like=pack_seed+'_').iloc[:, 0].tolist()
                prec_initialization+=df_prec_conf_pack_seed[:a]
                prec_learning+=df_prec_conf_pack_seed[a:b]
                prec_stabilization+=df_prec_conf_pack_seed[b:]

            matrix_conf_initialization.append(prec_initialization)
            matrix_conf_learning.append(prec_learning)
            matrix_conf_stabilization.append(prec_stabilization)

        return matrix_conf_initialization,matrix_conf_learning,matrix_conf_stabilization,conf_list

    if which_graph=='test_conf_prec_cost':

        def get_conf_from_column_name(column_names):
            
            n_ep_list,freq_list=[],[]
            for column_name in column_names:
                splited=column_name.split('_')
                n_ep_list.append(splited[3])
                freq_list.append(splited[4])
   
            return sorted(list(set(n_ep_list)), key=int),sorted(list(set(freq_list)), key=int)

        df_prec=pd.read_csv(path[1])
        df_prec = df_prec.filter(like=pack) # Solo columnas del pack
        df_prec = df_prec.filter(regex=prec_metric+"$") # Solo columnas del pack con metrica de prec indicada

        # Almacenar datos por configuracion para cada region
        matrix1,matrix2,matrix3=[],[],[]
        n_ep_list,freq_list=get_conf_from_column_name(df_prec.columns)

        for n_ep in tqdm(n_ep_list,desc="Data for test_prec_cost graphs"):
            n_ep_prec_initialization,n_ep_prec_learning,n_ep_prec_stabilization=[],[],[]

            for freq in freq_list:
                freq_prec_initialization,freq_prec_learning,freq_prec_stabilization=[],[],[]

                df_prec_conf=df_prec.filter(like='_'+n_ep+'_'+freq+'_')

                for pack_seed in df_limits['pack_seed']:
                    a=int(df_limits.loc[df_limits['pack_seed'] == pack_seed, 'a'])
                    b=int(df_limits.loc[df_limits['pack_seed'] == pack_seed, 'b'])

                    df_prec_conf_pack_seed= df_prec_conf.filter(like=pack_seed+'_').iloc[:, 0].tolist()
                    freq_prec_initialization+=df_prec_conf_pack_seed[:a]
                    freq_prec_learning+=df_prec_conf_pack_seed[a:b]
                    freq_prec_stabilization+=df_prec_conf_pack_seed[b:]

                n_ep_prec_initialization.append(freq_prec_initialization)
                n_ep_prec_learning.append(freq_prec_learning)
                n_ep_prec_stabilization.append(freq_prec_stabilization)

            # Escribir aqui las listas para que no pete la memoria
            matrix1.append(n_ep_prec_initialization)
            matrix2.append(n_ep_prec_learning)
            matrix3.append(n_ep_prec_stabilization)

        return matrix1,matrix2,matrix3,n_ep_list,freq_list

           
def main(pack,
         global_deg_metric='norm_worsening_to_improvement',local_deg_metric='reward_diff',
         prec_metric='relative_perc_criteria_best',
         limit_metric='from_first_last'):

    '''
    Genera las graficas a partir de los datos transformados a formato apropiado para ello.
    '''
    path0='experiments_intuition/results/SingleEnvAnalysis/data/pack/learning_regions.csv'
    path1='experiments_intuition/results/SingleEnvAnalysis/data/pack/deg_evolution.csv'
    path2='experiments_intuition/results/SingleEnvAnalysis/data/pack/train_prec.csv'
    path3='experiments_intuition/results/SingleEnvAnalysis/data/pack/test_prec.csv'
    path4='experiments_intuition/results/SingleEnvAnalysis/data/pack/test_cost.csv'

    # Cuadricula de grafica
    fig,axs=plt.subplots(4,3, figsize=(10,10),height_ratios=[0.05,0.02,0.05,0.3])
    plt.subplots_adjust(top=0.95,bottom=0.15,left=0.1,right=0.95, hspace=0.02,wspace=0.02)

    # 1) Distribucion de degradacion
    def desgradation_distribution(ax,title,deg_list,nombre=None):
        data = np.array(deg_list)
        kde = KernelDensity(bandwidth=0.1, kernel='gaussian')
        kde.fit(data[:, None])
        x = np.linspace(0, 1, 200)
        y_prob = np.exp(kde.score_samples(x[:, None]))

        ax.plot(x, y_prob, color='gray')
        ax.fill_between(x, 0, y_prob, color='gray', alpha=0.3)
        ax.axvline(np.median(data), color='black')
        ax.set_xlim(-0.1,1.1)
        ax.set_title(title)
        ax.legend(title=str(len(data)),frameon=False)
        ax.set_xlabel('degradation')
        if nombre==None:
            ax.set_yticklabels([])

    deg1,deg2,deg3=from_df_data_to_graph_data(
        [path0,path1],pack,'deg_distribution',
        global_deg_metric=global_deg_metric,local_deg_metric=local_deg_metric,
        limit_metric=limit_metric
    ) 
    desgradation_distribution(axs[0,0],'Initialization',deg1,nombre=True)
    desgradation_distribution(axs[0,1],'Learning',deg2)
    desgradation_distribution(axs[0,2],'Stabilization',deg3)

    max_y = max(axs[0,0].get_ylim()[1],axs[0,1].get_ylim()[1],axs[0,2].get_ylim()[1])
    for ax in axs[0,:]:
        ax.set_ylim(0, max_y)

    # 2) Train conf_prec
    def train_conf_precCI(ax,listas,nombres=None):
        for y, data in zip(range(len(listas)), listas):

            ax.hlines(y, np.percentile(data, 5), np.percentile(data, 95), color='black')
            ax.vlines(np.percentile(data, 5), y - 0.2, y + 0.2, color='black')
            ax.vlines(np.percentile(data, 95), y - 0.2, y + 0.2, color='black')
            ax.plot(np.median(data), y, 'o', color='black')

        ax.set_xlim(-0.1,1.1)
        ax.grid(axis='x', linestyle='--',alpha=0.4)
        if nombres!=None:
            ax.set_yticks(range(len(listas)), nombres)
            ax.set_ylabel('train n_ep')
        else:
            ax.set_yticks([])
        ax.set_xticklabels([])
        
    prec1,prec2,prec3,conf_list=from_df_data_to_graph_data(
        [path0,path2],pack,which_graph='train_conf_prec',
        prec_metric=prec_metric,limit_metric=limit_metric
    )
    train_conf_precCI(axs[2,0],prec1,conf_list)
    train_conf_precCI(axs[2,1],prec2)
    train_conf_precCI(axs[2,2],prec3)


    # 3) Test conf_prec_cost
    def test_conf_precCI_costColor(ax,prec_matrix,cost_matrix,n_ep_list,freq_list,nombres=None):
        # Fijar colores y marcadores
        def obtain_color_and_marker(value):
            if 0 <= value < 0.05:
                return "#006400", "o", 30
            elif 0.05 <= value < 0.1:
                return "#66c266", "o", 10
            elif 0.1 <= value < 0.15:
                return "#6dc48aa2", "^", 20
            elif 0.15 <= value < 0.2:
                return "#C7040436", "s", 10
            elif 0.2 <= value < 0.25:
                return "#C7040489", "s", 30
            else:
                return "#C70404", "*", 40

        legend_elements = [
            Line2D([0], [0], marker='o', color='w',
                markerfacecolor='#006400', markersize=8,
                label='0-5'),

            Line2D([0], [0], marker='o', color='w',
                markerfacecolor='#66c266', markersize=6,
                label='5-10'),

            Line2D([0], [0], marker='^', color='w',
                markerfacecolor='#6dc48aa2', markersize=10,
                label='10-15'),

            Line2D([0], [0], marker='s', color='w',
                markerfacecolor='#C7040436', markersize=6,
                label='15-20'),

            Line2D([0], [0], marker='s', color='w',
                markerfacecolor='#C7040489', markersize=8,
                label='20-25'),

            Line2D([0], [0], marker='*', color='w',
                markerfacecolor='#C70404', markersize=14,
                label='25-1')
        ]
        
        # Grafica
        current_height = 0
        segment_labels = []
        region_centers = []

        for i in range(len(n_ep_list)):
            
            region_start = current_height
            for j in range(len(freq_list)):
                datos = prec_matrix[i][j]
                datos_color = cost_matrix[i][j]
                        
                color, marcador, tamaño = obtain_color_and_marker(np.mean(datos_color))

                ax.hlines(current_height, np.percentile(datos, 5), np.percentile(datos, 95), color=color)
                ax.vlines([np.percentile(datos, 5), np.percentile(datos, 95)], current_height-0.2, current_height+0.2, color=color)
                ax.scatter(np.median(datos), current_height, color=color, marker=marcador, s=tamaño, zorder=3)
                
                segment_labels.append(freq_list[j])
                current_height += 1
            
            region_end = current_height - 1
            region_centers.append((region_start + region_end)/2)
            
            if i < 5: # Linea separadora de regiones
                ax.axhline(current_height-0.5, color='black', linewidth=1)

        ax.grid(axis='x', linestyle='--', alpha=0.5)
        ax.set_yticks(range(len(segment_labels)), segment_labels)

        if nombres!=None:
            ax.legend(handles=legend_elements,title="Mean val_cost_perc",loc='upper center',bbox_to_anchor=(1, -0.1), ncol=len(legend_elements),frameon=True)
            for center, name in zip(region_centers, n_ep_list):
                ax.text(-0.15, center, name,transform=ax.get_yaxis_transform(),ha='right', va='center')
            ax.set_ylabel('test (n_ep,freq)',labelpad=35)
        else:
            ax.set_yticklabels([])

        ax.set_xlim(-0.1,1.1)
        ax.set_xlabel("prec")
        ax.set_title("")
        ax.invert_yaxis()

    prec1,prec2,prec3,n_ep_list,freq_list=from_df_data_to_graph_data(
        [path0,path3],pack,which_graph='test_conf_prec_cost',cost_prec='prec',
        prec_metric=prec_metric,limit_metric=limit_metric
    )
    cost1,cost2,cost3,n_ep_list,freq_list=from_df_data_to_graph_data(
        [path0,path4],pack,which_graph='test_conf_prec_cost',cost_prec='cost',
        prec_metric=prec_metric,limit_metric=limit_metric
    )
    test_conf_precCI_costColor(axs[3,0],prec1,cost1,n_ep_list,freq_list,nombres=True)
    test_conf_precCI_costColor(axs[3,1],prec2,cost2,n_ep_list,freq_list)
    test_conf_precCI_costColor(axs[3,2],prec3,cost3,n_ep_list,freq_list)

    axs[1, 0].axis('off')
    axs[1, 1].axis('off')
    axs[1, 2].axis('off')

    plt.savefig('experiments_intuition/results/FinalGraph/'+pack+'_'+global_deg_metric+'_'+local_deg_metric+'_'+prec_metric+'.pdf')



# main('pack_PPO_BipedalWalker')
# main('pack_PPO_BipedalWalker',
#         global_deg_metric='best_last_deg',local_deg_metric='paired_diff_probpos')
main('pack_PPO_BipedalWalker',
        global_deg_metric='norm_from_mean_worsening_to_improvement',local_deg_metric='reward_diff')


