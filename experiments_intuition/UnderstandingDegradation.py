'''
Aprovechando los datos almacenados con Stable Baselines3 para Ant 
usando ejecucion secuencial y midiendo el tiempo en steps:

- Identificar semillas interesante: Moderate, Critical, Catastrophic
- Grafica de kernel density estimation por iteracion

NOTE: los cambios bruscos en el average episodic reward (ER) no son por outliers, la distribucion de los ER cambia significativamente.

TODO: (con datos obtenidos del cluster)
- Cuantos samples necesitamos como minimo para conocer de manera precisa cuando una politica domina a otra significativamente, i.e.,
para poder calcular de manera fiable el "stochastic dominance" del ER de dos politicas. Realmente queremos saber cuando cambia la 
mejor, y si el cambio es significativo.
- Mirar como de costosa es una validacion con ese tamaño de sample frente al tiempo disponible. Si no es grande, o es un porcentaje
aceptable, puede que definir una frecuencia adaptativa sea incoherente.
- En general, estaria bien estudiar la relacion entre: 
    1) tamaño de sample de validacion
    2) coste extra de validacion
    3) precision para determinar el "stochastic dominance"

'''
from sklearn.neighbors import KernelDensity
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
import sys, os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parent_dir )
from experiments.our_library import UtilsDataFrame


def learning_curve(x,env_name,seed,title):

    fig=plt.figure(figsize=[15,2.5])
    plt.subplots_adjust(left=0.1,bottom=0.2,right=0.94,top=0.82,wspace=0.39,hspace=0.2)
    ax=plt.subplot(111)
    ax.grid(True, which='both',linestyle='--', linewidth=0.8,alpha=0.2)

    df_test=pd.read_parquet(parent_dir+'/results/EnvironmentProcesses/'+str(env_name)+'/df_test_'+str(env_name)+'_seed'+str(seed)+'.parquet')
    df_train=pd.read_parquet(parent_dir+'/results/EnvironmentProcesses/'+str(env_name)+'/df_train_'+str(env_name)+'_seed'+str(seed)+'.parquet')

    # Dibujar curva sin test durante el proceso (ultima politica observada)
    y=[]
    x_policy=[]
    for timesteps in x:
            last_policy=df_train[df_train['n_train_timesteps']<=timesteps]['n_policy'].max()+1
            y.append(float(df_test[df_test['n_policy']==last_policy]['mean_reward']))
            x_policy.append(last_policy)
    plt.plot(x_policy, y, linewidth=1,color='black')

    plt.title(title+' degradation')
    ax.set_xlabel("Total iterations")
    ax.set_ylabel("$\widetilde{f}$ of the last policy")
    plt.savefig('experiments_intuition/results/UnderstandingDegradation/'+env_name+str(seed)+title+'_learning_curve.pdf')
    plt.show()
    plt.close()



def episodic_reward_KDE_per_iteration(env_name,seed,iterations):
    # Validation dataframe
    df_test=pd.read_parquet(parent_dir+'/results/EnvironmentProcesses/'+str(env_name)+'/df_test_'+str(env_name)+'_seed'+str(seed)+'.parquet')
    column_ep_test_rewards=[UtilsDataFrame.compress_decompress_list(i,compress=False) for i in df_test['ep_test_rewards']]
    df_test['ep_test_rewards']=column_ep_test_rewards

    # Normalizar episodic rewards
    def normalize_ep_rw(x):
        return (np.array(x)-min_ep_rw)/(max_ep_rw-min_ep_rw)
    
    max_ep_rw=np.max(np.array(column_ep_test_rewards))
    min_ep_rw=np.min(np.array(column_ep_test_rewards))
    df_test['ep_test_rewards_norm']=[normalize_ep_rw(i) for i in column_ep_test_rewards]

    # KDE per iteration
    fig=plt.figure(figsize=[20,5])
    plt.subplots_adjust(left=0.04,bottom=0.3,right=0.97,top=0.88,wspace=0.2,hspace=0.2)
    ax=plt.subplot(111)

    x_translate_list=[]
    x_translate=0
    x_d=np.linspace(0,1, 1000)
    avg_ep_rw=[]
    expected_ep_rw=[]
    for i in iterations:
        # KDE con los datos de la iteracion
        x=np.array(df_test['ep_test_rewards_norm'][i])
        kde = KernelDensity(bandwidth=0.05, kernel='gaussian')
        kde.fit(x[:, None])
        y_prob = np.exp(kde.score_samples(x_d[:, None]))

        # Dibujar curva de densidad
        if i== iterations[0]:
            plt.plot(np.full_like(x, -0.01)+x_translate,x, '_', markeredgewidth=1,color='black',alpha=0.5,label='Sample of 100 ER\n(initialized from de same 100 states)')
            plt.fill_betweenx( x_d,y_prob+x_translate,min(y_prob)+x_translate,color='blue',alpha=0.5,label='KDE')
        else:
            plt.plot(np.full_like(x, -0.01)+x_translate,x, '_', markeredgewidth=1,color='black',alpha=0.5)
            plt.fill_betweenx( x_d,y_prob+x_translate,min(y_prob)+x_translate,color='blue',alpha=0.5)
        x_translate_list.append(x_translate)
        x_translate+=max(y_prob)*2

        avg_ep_rw.append(np.mean(x))
        expected_ep_rw.append(simpson(x_d*y_prob, x_d))

    # Curvas de medias y expected values
    plt.plot(x_translate_list,avg_ep_rw,color='red',label='Average ER')
    # plt.plot(x_translate_list,expected_ep_rw,color='green',label='Expected ER')
    

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3),ncol=4)
    plt.xticks(ticks=x_translate_list, labels=iterations,rotation=60)
    plt.ylabel('Normalized Episodic Reward (ER)')
    plt.xlabel('Policy')
    plt.savefig('experiments_intuition/results/UnderstandingDegradation/'+env_name+str(seed)+'_'+str(min(iterations))+'_'+str(max(iterations))+'_'+str(iterations[1]-iterations[0])+'.pdf')
    plt.show()


learning_curve(list(range(2048, 3200000+2048, 2048)),'Ant',6,'Moderate')
episodic_reward_KDE_per_iteration('Ant',6,range(0,1560,25))
episodic_reward_KDE_per_iteration('Ant',6,range(0,1560,50))
episodic_reward_KDE_per_iteration('Ant',6,range(0,1560,100))
 
learning_curve(list(range(2048, 3200000+2048, 2048)),'Ant',1,'Critical')
episodic_reward_KDE_per_iteration('Ant',1,range(600,810,10))
episodic_reward_KDE_per_iteration('Ant',1,range(600,805,5))
episodic_reward_KDE_per_iteration('Ant',1,range(600,636,1))
episodic_reward_KDE_per_iteration('Ant',1,range(850,952,2))
episodic_reward_KDE_per_iteration('Ant',1,range(1500,1562,2))
episodic_reward_KDE_per_iteration('Ant',1,range(1540,1561,1))
episodic_reward_KDE_per_iteration('Ant',1,range(0,1560,25))
episodic_reward_KDE_per_iteration('Ant',1,range(0,1560,50))
episodic_reward_KDE_per_iteration('Ant',1,range(0,1560,100))

learning_curve(list(range(2048, 3200000+2048, 2048)),'Ant',3,'Catastrophic')
episodic_reward_KDE_per_iteration('Ant',3,range(700,805,5))
episodic_reward_KDE_per_iteration('Ant',3,range(715,741,1))
episodic_reward_KDE_per_iteration('Ant',3,range(1000,1105,5))
episodic_reward_KDE_per_iteration('Ant',3,range(1020,1036,1))
episodic_reward_KDE_per_iteration('Ant',3,range(0,1560,25))
episodic_reward_KDE_per_iteration('Ant',3,range(0,1560,50))
episodic_reward_KDE_per_iteration('Ant',3,range(0,1560,100))

episodic_reward_KDE_per_iteration('Ant',3,[299,399,599])


