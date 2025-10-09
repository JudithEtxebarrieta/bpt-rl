'''
Script para analizar el numero de iteraciones a considerar para ejecutar PPO en el cluster. 
Queremos conocer el numero de iteraciones por entorno que permiten a PPO converger y un poco mas.
Usando los datos de train almacenados en df_traj.csv, se puede estudiar la evolucion de ciertas metricas
para detectar la convergencia, e.g. entropy_loss.
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def graph_conv(envs,seeds):

    # Funciones para estimar de manera automatica el punto de convergencia
    # (percentil de diferencia de metrica en ventana menor que una tolerancia/umbral)
    def descendant(list_y):
        descen=[]
        for i in range(len(y)-1):
            descen.append([list_y[i]-list_y[i+1] if list_y[i]-list_y[i+1]>0 else 0][0])

        return descen
    
    def var_window(list_y,w_size=50,tol=0.01):

        desc=descendant(list_y)
        w_perc=[]
        for i in range(w_size,len(list_y)):
            w_perc.append(np.percentile(desc[i-w_size:i],90))
        
        cuantos=0
        conv=None
        for i in range(len(w_perc)):
            if round(w_perc[i],2)<=tol:
                cuantos+=1
            if cuantos==w_size:
                conv=i+w_size
        return conv

    # Evolucion de metrica
    fig,axes=plt.subplots(len(seeds),len(envs),figsize=(len(envs)*4, len(seeds)*3))
    plt.subplots_adjust(left=0.08,bottom=0.15,right=0.97,top=0.92,wspace=0.2,hspace=0.07)

    for i in range(len(envs)):
        for j in range(len(seeds)):

            # Leer base de datos que tiene las metricas calculadas durante train (metricas de convergencia)
            df_train=pd.read_csv('_bender/project_SB3/data/PPO_'+envs[i]+'_seed'+str(seeds[j])+'_CONV/df_traj.csv')

            y=df_train['entropy_loss'].tolist()# tenemos estos datos almacenados: policy_loss,value_loss,entropy_loss,policy_gradient_loss,KL_div,explained_variance,log_std

            conv=var_window(y)
            axes[i+len(envs)*j].plot(range(len(y)),y)
            if conv is not None:
                axes[i+len(envs)*j].axvline(x=conv,color='red')

            if j==0:
                axes[i+len(envs)*j].set_title(envs[i])
            if i==0:
                axes[i+len(envs)*j].set_ylabel("entropy_loss")
            if j==len(seeds)-1:
                axes[i+len(envs)*j].set_xlabel("Learning iteration")

            axes[i+len(envs)*j].grid(True)

    plt.savefig('experiments_intuition/results/ConvergenceAnalysis/conv.pdf')
    plt.show()


# Main
envs=['HalfCheetah','Walker2d','Ant','Humanoid','HumanoidStandup']
seeds=[1]
graph_conv(envs,seeds)