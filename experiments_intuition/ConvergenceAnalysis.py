'''
Script para analizar el numero de iteraciones a considerar para ejecutar PPO en el cluster. 
Queremos conocer el numero de iteraciones por entorno que permiten a PPO converger y un poco mas.
Usando los datos de train almacenados en df_traj.csv, se puede estudiar la evolucion de ciertas metricas
para detectar la convergencia, e.g. entropy_loss.
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def graph_conv(env):

    # Leer base de datos que tiene las metricas calculadas durante train (metricas de convergencia)
    df_train=pd.read_csv('_bender/project_SB3/data/PPO_'+env+'_seed1_CONV/df_traj.csv')

    # Funciones para estimar de manera automatica el punto de convergencia
    # (percentil de diferencia de metrica en ventana menor que una tolerancia/umbral)
    def descendant(list_y):
        descen=[]
        for i in range(len(y)-1):
            descen.append([abs(list_y[i]-list_y[i+1]) ][0])

        return descen
    
    def var_window(list_y,w_size=50,tol=0.01):

        desc=descendant(list_y)
        w_perc=[]
        for i in range(w_size,len(list_y)):
            w_perc.append(np.percentile(desc[i-w_size:i],75))
        
        cuantos=0
        conv=None
        for i in range(len(w_perc)):
            if round(w_perc[i],2)<=tol:
                cuantos+=1
            if cuantos==w_size:
                conv=i+w_size

        return conv

    # Evolucion de metrica
    plt.figure(figsize=(8, 5))

    y=df_train['entropy_loss'].tolist()# tenemos estos datos almacenados: entropy_loss,policy_gradient_loss,KL_div,explained_variance,log_std
    conv=var_window(y)
    plt.plot(range(len(y)),y, label="entropy_loss")
    if conv is not None:
        plt.axvline(x=conv,color='red')

    plt.ylabel("Convergence metric")
    plt.xlabel("Learning iteration")
    plt.title("")
    plt.legend()
    plt.grid(True)
    plt.savefig('experiments_intuition/results/ConvergenceAnalysis/conv_'+env+'.pdf')


# Main
envs=['HalfCheetah','Walker2d','Ant','Humanoid','HumanoidStandup']
for env in envs:
    graph_conv(env)