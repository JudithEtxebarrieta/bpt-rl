from Main import *
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def from_df_data_to_graph_data(path,pack,
                                  which_graph=None,
                                  global_deg_metric=None,local_deg_metric=None,
                                  prec_metric=None,limit_metric=None,
                                  train_conf=None,test_conf=None):
    
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
                if 'cost' not in splited[3] and splited[3] not in ['','relative']: # TODO: ahora mismo en test_prec.csv hay algunas columnas mal guardadas, tendria que volver a generar ese csv y quitar la ultimas 2 condicion de este if
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

    if which_graph=='how_times_best':
        # Quedarnos unicamente con los datos necesarios
        df_last=pd.read_csv(path[1])
        df_last = df_last.filter(regex=r''+pack) # Solo columnas asociadas al pack

        df_train=pd.read_csv(path[2])
        df_train = df_train.filter(regex=r''+pack+'.*_'+str(train_conf)+'$') # Solo columnas asociadas al pack y la configuracion indicada

        df_test=pd.read_csv(path[3])
        df_test = df_test.filter(regex=r''+pack+'.*_'+str(test_conf)+'$') # Solo columnas asociadas al pack y la configuracion indicada

        # Contar las veces que es cada par de criterios mejor por regiones (sin empates)
        last_train1,last_train2,last_train3=[0,0],[0,0],[0,0]
        last_test1,last_test2,last_test3=[0,0],[0,0],[0,0]
        train_test1,train_test2,train_test3=[0,0],[0,0],[0,0]
        for pack_seed in df_limits['pack_seed']:

            def update_times_best_with_new_seed(truth1,truth2,a,b,old1,old2,old3):
                truth1=np.array(truth1)
                truth2=np.array(truth2)
                olds=(np.array(old1),np.array(old2),np.array(old3))

                news=[np.array([np.sum(truth1[:a]>truth2[:a]), np.sum(truth1[:a]<truth2[:a])]),
                        np.array([np.sum(truth1[a:b]>truth2[a:b]), np.sum(truth1[a:b]<truth2[a:b])]),
                        np.array([np.sum(truth1[b:]>truth2[b:]), np.sum(truth1[b:]<truth2[b:])])
                        ]

                return [old+new for old,new in zip(olds, news)]
            
            a=int(df_limits.loc[df_limits['pack_seed'] == pack_seed, 'a'])
            b=int(df_limits.loc[df_limits['pack_seed'] == pack_seed, 'b'])

            truth_last=df_last.filter(like=pack_seed).iloc[:, 0].tolist()
            truth_train=df_train.filter(like=pack_seed+'_'+str(train_conf)).iloc[:, 0].tolist()
            last_train1,last_train2,last_train3 = update_times_best_with_new_seed(truth_last,truth_train,a,b,last_train1,last_train2,last_train3)

            truth_test=df_test.filter(like=pack_seed+'_'+str(test_conf)).iloc[:, 0].tolist()
            last_test1,last_test2,last_test3 = update_times_best_with_new_seed(truth_last,truth_test,a,b,last_test1,last_test2,last_test3)

            truth_test=df_test.filter(like=pack_seed+'_'+str(test_conf)).iloc[:, 0].tolist()
            train_test1,train_test2,train_test3 = update_times_best_with_new_seed(truth_train,truth_test,a,b,train_test1,train_test2,train_test3)

        # Pasar de numero de veces a procentage
        matrix1=np.array([last_train1,last_test1,train_test1])/df_limits['a'].sum()
        matrix2=np.array([last_train2,last_test2,train_test2])/(df_limits['b'] - df_limits['a']).sum()
        matrix3=np.array([last_train3,last_test3,train_test3])/(df_limits['T'] - df_limits['b']).sum()

        return matrix1,matrix2,matrix3

    if which_graph=='in_which_deg_best':

        # Quedarnos unicamente con los datos necesarios
        df_last=pd.read_csv(path[1])
        df_last = df_last.filter(regex=r''+pack) # Solo columnas asociadas al pack

        df_train=pd.read_csv(path[2])
        df_train = df_train.filter(regex=r''+pack+'.*_'+str(train_conf)+'$') # Solo columnas asociadas al pack y la configuracion indicada

        df_test=pd.read_csv(path[3])
        df_test = df_test.filter(regex=r''+pack+'.*_'+str(test_conf)+'$') # Solo columnas asociadas al pack y la configuracion indicada

        df_deg=pd.read_csv(path[4])
        df_deg = df_deg.filter(like=pack) # Solo columnas del pack
        df_deg = df_deg.filter(regex=global_deg_metric+'_'+local_deg_metric+"$") # Solo columnas del pack con deg indicada

        # Acumular las degradaciones en que cada uno de los criterios de cada par es mejor 
        last_train1,last_train2,last_train3=[[],[]],[[],[]],[[],[]]
        last_test1,last_test2,last_test3=[[],[]],[[],[]],[[],[]]
        train_test1,train_test2,train_test3=[[],[]],[[],[]],[[],[]]
        for pack_seed in df_limits['pack_seed']:

            def update_deg_best_with_new_seed(deg,truth1,truth2,a,b,old1,old2,old3):
                truth1=np.array(truth1)
                truth2=np.array(truth2)
                deg=np.array(deg)
                olds=[old1,old2,old3]

                news=[[deg[:a][truth1[:a] > truth2[:a]],deg[:a][truth1[:a] < truth2[:a]]],
                [deg[a:b][truth1[a:b] > truth2[a:b]],deg[a:b][truth1[a:b] < truth2[a:b]]],
                [deg[b:][truth1[b:] > truth2[b:]],deg[b:][truth1[b:] < truth2[b:]]]
                ]

                return [[list(sub_old) + list(sub_new) for sub_old, sub_new in zip(old_i, new_i)] for old_i, new_i in zip(olds, news)]
            
            a=int(df_limits.loc[df_limits['pack_seed'] == pack_seed, 'a'])
            b=int(df_limits.loc[df_limits['pack_seed'] == pack_seed, 'b'])
            deg=df_deg.filter(like=pack_seed).iloc[:, 0].tolist()


            truth_last=df_last.filter(like=pack_seed).iloc[:, 0].tolist()
            truth_train=df_train.filter(like=pack_seed+'_'+str(train_conf)).iloc[:, 0].tolist()
            last_train1,last_train2,last_train3 = update_deg_best_with_new_seed(deg,truth_last,truth_train,a,b,last_train1,last_train2,last_train3)

            truth_test=df_test.filter(like=pack_seed+'_'+str(test_conf)).iloc[:, 0].tolist()
            last_test1,last_test2,last_test3 = update_deg_best_with_new_seed(deg,truth_last,truth_test,a,b,last_test1,last_test2,last_test3)

            truth_test=df_test.filter(like=pack_seed+'_'+str(test_conf)).iloc[:, 0].tolist()
            train_test1,train_test2,train_test3 = update_deg_best_with_new_seed(deg,truth_train,truth_test,a,b,train_test1,train_test2,train_test3)

        return [last_train1,last_test1,train_test1], [last_train2,last_test2,train_test2], [last_train3,last_test3,train_test3]

    if which_graph=='with_what_prec_diff_best':
        # Quedarnos unicamente con los datos necesarios
        df_last=pd.read_csv(path[1])
        df_last = df_last.filter(regex=r''+pack) # Solo columnas asociadas al pack

        df_train=pd.read_csv(path[2])
        df_train = df_train.filter(regex=r''+pack+'.*_'+str(train_conf)+'$') # Solo columnas asociadas al pack y la configuracion indicada

        df_test=pd.read_csv(path[3])
        df_test = df_test.filter(regex=r''+pack+'.*_'+str(test_conf)+'$') # Solo columnas asociadas al pack y la configuracion indicada

        df_prec_last=pd.read_csv(path[4])
        df_prec_last = df_prec_last.filter(regex=r''+pack+'.*_'+prec_metric+'$') # Solo columnas del pack con metrica de prec indicada

        df_prec_train=pd.read_csv(path[5])
        df_prec_train = df_prec_train.filter(regex=r''+pack+'.*_'+str(train_conf)+'_'+prec_metric+'$') # Solo columnas del pack con metrica de prec indicada

        df_prec_test=pd.read_csv(path[6])
        df_prec_test = df_prec_test.filter(regex=r''+pack+'.*_'+str(test_conf)+'_'+prec_metric+'$') # Solo columnas del pack con metrica de prec indicada

        # Acumular las precisiones en que cada uno de los criterios de cada par es mejor 
        last_train1,last_train2,last_train3=[[[],[]],[[],[]]],[[[],[]],[[],[]]],[[[],[]],[[],[]]]
        last_test1,last_test2,last_test3=[[[],[]],[[],[]]],[[[],[]],[[],[]]],[[[],[]],[[],[]]]
        train_test1,train_test2,train_test3=[[[],[]],[[],[]]],[[[],[]],[[],[]]],[[[],[]],[[],[]]]
        for pack_seed in df_limits['pack_seed']:

            def update_prec_best_with_new_seed(prec1,prec2,truth1,truth2,a,b,old1,old2,old3):
                truth1=np.array(truth1)
                truth2=np.array(truth2)
                prec1=np.array(prec1)
                prec2=np.array(prec2)
                olds=[old1,old2,old3]

                news=[[[prec1[:a][truth1[:a] > truth2[:a]],prec2[:a][truth1[:a] > truth2[:a]]],[prec2[:a][truth1[:a] < truth2[:a]],prec1[:a][truth1[:a] < truth2[:a]]]],
                [[prec1[a:b][truth1[a:b] > truth2[a:b]],prec2[a:b][truth1[a:b] > truth2[a:b]]],[prec2[a:b][truth1[a:b] < truth2[a:b]],prec1[a:b][truth1[a:b] < truth2[a:b]]]],
                [[prec1[b:][truth1[b:] > truth2[b:]],prec2[b:][truth1[b:] > truth2[b:]]],[prec2[b:][truth1[b:] < truth2[b:]],prec1[b:][truth1[b:] < truth2[b:]]]]
                ]

                return [[[list(x) + list(y) 
                    for x,y in zip(a_ij, b_ij)] 
                    for a_ij, b_ij in zip(a_i, b_i)] 
                    for a_i, b_i in zip(olds, news)]
            
            a=int(df_limits.loc[df_limits['pack_seed'] == pack_seed, 'a'])
            b=int(df_limits.loc[df_limits['pack_seed'] == pack_seed, 'b'])
            prec_last=df_prec_last.filter(like=pack_seed+'_'+prec_metric).iloc[:, 0].tolist()
            prec_train=df_prec_train.filter(like=pack_seed+'_'+str(train_conf)+'_'+prec_metric).iloc[:, 0].tolist()
            prec_test=df_prec_test.filter(like=pack_seed+'_'+str(test_conf)+'_'+prec_metric).iloc[:, 0].tolist()



            truth_last=df_last.filter(like=pack_seed).iloc[:, 0].tolist()
            truth_train=df_train.filter(like=pack_seed+'_'+str(train_conf)).iloc[:, 0].tolist()
            last_train1,last_train2,last_train3 = update_prec_best_with_new_seed(prec_last,prec_train,truth_last,truth_train,a,b,last_train1,last_train2,last_train3)

            truth_test=df_test.filter(like=pack_seed+'_'+str(test_conf)).iloc[:, 0].tolist()
            last_test1,last_test2,last_test3 = update_prec_best_with_new_seed(prec_last,prec_test,truth_last,truth_test,a,b,last_test1,last_test2,last_test3)

            truth_test=df_test.filter(like=pack_seed+'_'+str(test_conf)).iloc[:, 0].tolist()
            train_test1,train_test2,train_test3 = update_prec_best_with_new_seed(prec_train,prec_test,truth_train,truth_test,a,b,train_test1,train_test2,train_test3)

        return [last_train1,last_test1,train_test1], [last_train2,last_test2,train_test2], [last_train3,last_test3,train_test3]

        
def main1(pack,
         global_deg_metric='norm_worsening_to_improvement',local_deg_metric='reward_diff',
         prec_metric='relative_perc_criteria_best',
         limit_metric='from_first_last'):

    '''
    Genera primera grafica principal a partir de los datos transformados a formato apropiado para ello.
    - Distribuciones de degradacion por region de aprendizaje
    - Precision y coste de seleccion para diferentes configuraciones de los criterios train y test
    (esto sirve como analisis de sensibilidad y configuracion optima)
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
                label='>25')
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
        [path0,path3],pack,which_graph='test_conf_prec_cost',
        prec_metric=prec_metric,limit_metric=limit_metric
    )
    cost1,cost2,cost3,n_ep_list,freq_list=from_df_data_to_graph_data(
        [path0,path4],pack,which_graph='test_conf_prec_cost',
        prec_metric=prec_metric,limit_metric=limit_metric
    )
    test_conf_precCI_costColor(axs[3,0],prec1,cost1,n_ep_list,freq_list,nombres=True)
    test_conf_precCI_costColor(axs[3,1],prec2,cost2,n_ep_list,freq_list)
    test_conf_precCI_costColor(axs[3,2],prec3,cost3,n_ep_list,freq_list)

    axs[1, 0].axis('off')
    axs[1, 1].axis('off')
    axs[1, 2].axis('off')

    plt.savefig('experiments_intuition/results/FinalGraph/'+pack+'_'+global_deg_metric+'_'+local_deg_metric+'_'+prec_metric+'.pdf')


def main2(pack,
         global_deg_metric='norm_worsening_to_improvement',local_deg_metric='reward_diff',
         prec_metric='relative_perc_criteria_best',
         limit_metric='from_first_last',
         train_conf=100,test_conf='0.2cost'):

    '''
    Genera la segunda grafica principal a partir de los datos transformados a formato apropiado para ello.
    - Comparacion de criterios de dos en dos
    - Frecuencia con que cada criterio es mejor
    - Distribucion de degradacion en que cada criterio es mejor
    - Precision de seleccion cuando cada criterio es el mejor y con que diferencia al otro es mejor
    '''
    path0='experiments_intuition/results/SingleEnvAnalysis/data/pack/learning_regions.csv'
    path1='experiments_intuition/results/SingleEnvAnalysis/data/pack/df_last_truth.csv'
    path2='experiments_intuition/results/SingleEnvAnalysis/data/pack/df_train_truth.csv'
    path3='experiments_intuition/results/SingleEnvAnalysis/data/pack/df_test_truth.csv'
    path4='experiments_intuition/results/SingleEnvAnalysis/data/pack/deg_evolution.csv'
    path5='experiments_intuition/results/SingleEnvAnalysis/data/pack/last_prec.csv'
    path6='experiments_intuition/results/SingleEnvAnalysis/data/pack/train_prec.csv'
    path7='experiments_intuition/results/SingleEnvAnalysis/data/pack/test_prec.csv'

    # Cuadricula de grafica
    fig,axs=plt.subplots(3,3, figsize=(10,6),height_ratios=[0.03,0.1,0.1])
    plt.subplots_adjust(top=0.95,bottom=0.15,left=0.1,right=0.95, hspace=0.02,wspace=0.02)

    # 1) ¿Cuantas veces es el mejor cada criterio comparados de dos en dos?
    def how_many_times_better(ax,data,title,nombre=None):
        colors = [['blue', 'orange'],['blue', 'green'],['orange', 'green']]

        for i, ((left_val, right_val), (color_left, color_right)) in enumerate(zip(data, colors)):
            ax.barh(y=i,width=left_val,left=0,color=color_left,height=1)
            ax.barh(y=i,width=right_val,left=1 - right_val,color=color_right,height=1)

        ax.set_xlim(-0.05,1.05)
        ax.set_ylim(-0.5, len(data)-0.5)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels([])
        if nombre:
            ax.set_yticklabels(['Last vs Train','Last vs Test','Train vs Test'],fontsize=8)
        ax.set_xticklabels([])
        ax.set_title(title)

    matrix1,matrix2,matrix3=from_df_data_to_graph_data([path0,path1,path2,path3],pack,
                                                       which_graph='how_times_best',
                                                       limit_metric=limit_metric,
                                                       train_conf=train_conf,test_conf=test_conf)

    how_many_times_better(axs[0,0],matrix1,'Initialization',nombre=True)
    how_many_times_better(axs[0,1],matrix2,'Learning')
    how_many_times_better(axs[0,2],matrix3,'Stabilization')

    # 2) ¿En que degradaciones es mejor cada criterio?
    def in_which_deg_best(ax,data,nombre=None):
        colors = [['blue', 'orange'],['blue', 'green'],['orange', 'green']]
        color_marker_map = {
            'blue': 'o',    # círculo
            'orange': '^',  # triángulo
            'green': 's'    # cuadrado
        }
        legend_elements = [
            Line2D([0], [0], marker='o', color='blue', label='Last', markersize=6, linestyle=''),
            Line2D([0], [0], marker='^', color='orange', label='Train', markersize=6, linestyle=''),
            Line2D([0], [0], marker='s', color='green', label='Test', markersize=6, linestyle='')
        ]


        y_spacing = 0.5
        kde_height=0.4

        for i, (pair, pair_colors) in enumerate(zip(data, colors)):
            for j, (sublist, color) in enumerate(zip(pair, pair_colors)):
                kde = gaussian_kde(sublist)
                x = np.linspace(0, 1, 200)
                y = kde(x) / kde(x).max() * kde_height + i*y_spacing  # Reescalamos la densidad vertical para que no se superpongan
                ax.plot(x, y, color=color)
                ax.fill_between(x, i*y_spacing, y, color=color, alpha=0.3)

                y_median = kde(np.median(sublist))[0] / kde(x).max() * kde_height + i*y_spacing
                ax.vlines(np.median(sublist),i*y_spacing,y_median,color=color)

                marker = color_marker_map.get(color, 'o')
                ax.plot(np.median(sublist), y_median, marker=marker, color=color, markersize=5)

        if nombre:
            ax.set_ylabel("In which degradations is the best?",fontsize=8)
            ax.legend(handles=legend_elements, loc='upper center',
                bbox_to_anchor=(0.5, -1.2), ncol=3, frameon=False)
        ax.set_yticks([i*y_spacing  for i in range(len(data))])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xlim(-0.05,1.05)


        

    matrix1,matrix2,matrix3=from_df_data_to_graph_data([path0,path1,path2,path3,path4],pack,
                                                       which_graph='in_which_deg_best',
                                                       limit_metric=limit_metric,
                                                       train_conf=train_conf,test_conf=test_conf,
                                                       global_deg_metric=global_deg_metric,local_deg_metric=local_deg_metric
                                                       )
    in_which_deg_best(axs[1,0],matrix1,nombre=True)
    in_which_deg_best(axs[1,1],matrix2)
    in_which_deg_best(axs[1,2],matrix3)

    # 3) ¿Cual es la precision de seleccion del criterio cuando es mejor?
    def with_what_prec_diff_best(ax,data,nombre=None):
        colors = [
                    [["#0000FF", "#FADDBB"], ["#FF8800", "#BABAFF"]],
                    [["#0000FF", "#C7EBC7"], ["#009900", "#BABAFF"]],
                    [["#FF8800", "#C7EBC7"], ["#009900", "#FADDBB"]]
                ]
        color_marker_map = {
            "#0000FF": 'o',   # azul → círculo
            "#FF8800": '^',   # naranja → triángulo
            "#009900": 's',    # verde → cuadrado
            "#FADDBB": '^', "#C7EBC7": 's', "#BABAFF": 'o'

        }
        
        region_spacing = 0.9
        level_spacing = 0.35      # separación entre subniveles
        inner_offset = 0.06       # separación pequeña entre los dos segmentos
        cap_height = 0.04         # tamaño de los topes verticales

        # Dibujar los intervalos
        for i, (region, region_colors) in enumerate(zip(data, colors)):
            base_y = i * region_spacing
            for j, (sublist, sub_colors) in enumerate(zip(region, region_colors)):
                base_level_y = base_y + j * level_spacing
                for k, (subsubdata, color) in enumerate(zip(sublist, sub_colors)):
                    offset = inner_offset if k == 0 else -inner_offset
                    y_pos = base_level_y + offset

                    ax.hlines(y_pos, np.percentile(subsubdata, 5), np.percentile(subsubdata, 95), color=color)
                    ax.vlines(np.percentile(subsubdata, 5),  y_pos - cap_height, y_pos + cap_height, color=color)
                    ax.vlines(np.percentile(subsubdata, 95), y_pos - cap_height, y_pos + cap_height, color=color)

                    marker = color_marker_map.get(color, 'o')  # default círculo si color no mapeado
                    ax.plot(np.median(subsubdata), y_pos, marker=marker, color=color, markersize=5)


        for i in range(1, len(data)):
            ax.axhline(i * region_spacing - level_spacing / 2 -0.1, color='black', linestyle='-',linewidth=0.5)

        region_centers = [i * region_spacing + level_spacing / 2 for i in range(len(data))]
        ax.set_yticks(region_centers)
        if nombre:
            ax.set_ylabel("How precise is it when it is the best\nand how does it differ from the other?", fontsize=8)
        ax.set_yticklabels([])
        ax.set_xlim(-0.05,1.05)
        ax.grid(axis='x', linestyle='--', color='gray', linewidth=0.8)

    matrix1,matrix2,matrix3=from_df_data_to_graph_data([path0,path1,path2,path3,path5,path6,path7],pack,
                                                       which_graph='with_what_prec_diff_best',
                                                       limit_metric=limit_metric,
                                                       train_conf=train_conf,test_conf=test_conf,
                                                       prec_metric=prec_metric
                                                       )

    with_what_prec_diff_best(axs[2,0],matrix1,nombre=True)
    with_what_prec_diff_best(axs[2,1],matrix2)
    with_what_prec_diff_best(axs[2,2],matrix3)

    
    plt.savefig('experiments_intuition/results/FinalGraph/2'+pack+'_'+global_deg_metric+'_'+local_deg_metric+'_'+prec_metric+'.pdf')


# Framework experimental, parte 1: degradacion y configuracion optima (prec vs cost)
main1('pack_PPO_BipedalWalker')
main1('pack_PPO_BipedalWalker',
        global_deg_metric='best_last_deg',local_deg_metric='paired_diff_probpos')
main1('pack_PPO_BipedalWalker',
        global_deg_metric='norm_from_mean_worsening_to_improvement',local_deg_metric='reward_diff')


# Framewrok experimental, parte 2: comparacion de criterios last-train-test
main2('pack_PPO_BipedalWalker',
         global_deg_metric='norm_from_mean_worsening_to_improvement',local_deg_metric='reward_diff',
         prec_metric='relative_perc_criteria_best',
         limit_metric='from_first_last',
         train_conf=100,test_conf='0.25cost')
