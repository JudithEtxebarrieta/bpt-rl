
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.colors as mcolors
from tqdm import tqdm
from our_library import  UtilsDataFrame, UtilsFigure

def validation_ep_selection(ep_test,n_test_ep,test_type='reward'):
    list_output=[]
    list_seeds=range(len(ep_test))
    for i in list_seeds:
        random.seed(i)
        if test_type=='reward':
            list_output.append(np.mean(random.sample(ep_test,n_test_ep)))
        if test_type=='len':
            list_output.append(sum(random.sample(ep_test,n_test_ep)))
        

    return list_output

def degradation(x,env_name,seed,title):

    # Para guardar los resultados
    output_path='results/MAEB/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    fig=plt.figure(figsize=[3.5,2.5])
    plt.subplots_adjust(left=0.20,bottom=0.2,right=0.94,top=0.82,wspace=0.39,hspace=0.2)
    plt.rc('font', family='serif')
    plt.rc('text', usetex=True)
    plt.rcParams['text.latex.preamble'] = r'\boldmath'

    ax=plt.subplot(111)
    ax.grid(True, which='both',linestyle='--', linewidth=0.8,alpha=0.2)
    
    df_test=pd.read_parquet('results/EnvironmentProcesses/'+str(env_name)+'/df_test_'+str(env_name)+'_seed'+str(seed)+'.parquet')
    df_train=pd.read_parquet('results/EnvironmentProcesses/'+str(env_name)+'/df_train_'+str(env_name)+'_seed'+str(seed)+'.parquet')

    y_matrix=[]
    labels=[]
    colors=[]

    # Dibujar curva sin test durante el proceso (ultima politica observada)
    y=[]
    for timesteps in x:
            last_policy=df_train[df_train['n_train_timesteps']<=timesteps]['n_policy'].max()+1
            y.append(float(df_test[df_test['n_policy']==last_policy]['mean_reward']))
    plt.plot(x, y, linewidth=1,color='black',label='None (last policy)')
    y_matrix.append(y)
    labels.append('None (last policy)')
    colors.append('black')

    plt.title(title,fontsize=14)
    ax.set_xlabel("Total interaction steps",fontsize=13)
    ax.set_ylabel("$\widetilde{J}$ of the last policy",fontsize=13)
    plt.yticks(rotation=90,fontsize=12)
    plt.xticks(fontsize=12)
    plt.ylim([0,3600])
    plt.savefig(output_path+'/degradation'+str(seed)+'.pdf')
    plt.show()
    plt.close()

def learning_curves_test_reward(x,list_n_test_ep,env_name,seed,list_eval_freq=[1]):

    # Para guardar los resultados
    output_path='results/MAEB/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    eval_freq_with_train=False

    df_test=pd.read_parquet('results/EnvironmentProcesses/'+str(env_name)+'/df_test_'+str(env_name)+'_seed'+str(seed)+'.parquet')
    df_train=pd.read_parquet('results/EnvironmentProcesses/'+str(env_name)+'/df_train_'+str(env_name)+'_seed'+str(seed)+'.parquet')
    df_test=df_test[df_test['n_policy']<=max(x)//2048]
    df_train=df_train[df_train['n_policy']<=max(x)//2048-1]

    n_policies=list(df_test['n_policy'])


    # Dibujar curvas
    fig=plt.figure(figsize=[5,3])
    plt.subplots_adjust(left=0.18,bottom=0.2,right=0.94,top=0.82,wspace=0.39,hspace=0.2)

    plt.rc('font', family='serif')
    plt.rc('text', usetex=True)
    plt.rcParams['text.latex.preamble'] = r'\boldmath'

    ax=plt.subplot(111)
    ax.grid(True, which='both',linestyle='--', linewidth=0.8,alpha=0.2)

    # Dibujar curva sin test durante el proceso (ultima politica observada)
    y=[]
    for timesteps in x:
            last_policy=df_train[df_train['n_train_timesteps']<=timesteps]['n_policy'].max()+1
            y.append(df_test[df_test['n_policy']==last_policy]['mean_reward'])
    plt.plot(x, y, linewidth=1,color='black',label='None (last policy)')

    # Resto de curvas
    if len(list_n_test_ep)>1:
        for_list=list_n_test_ep
    else:
        for_list=list_eval_freq
    
    for i in tqdm(for_list):

        
        if len(list_n_test_ep)>1:
            n_test_ep=i
            eval_freq=list_eval_freq[0]
            label=n_test_ep
        else:
            eval_freq=i
            n_test_ep=list_n_test_ep[0]
            label=eval_freq


        current_mean_reward=[]
        current_n_test_timesteps=[]
        cumulative_n_test_timesteps=[0]*100


        for policy in n_policies:
            
            if type(eval_freq) is int:
                condition=policy%eval_freq==0
            else:
                label,change=eval_freq
                condition=change[policy-1]>0
                eval_freq_with_train=True

            if condition:
                ep_test_rewards_compressed=list(df_test[df_test['n_policy']==policy]['ep_test_rewards'])
                ep_test_len_compressed=list(df_test[df_test['n_policy']==policy]['ep_test_len'])

                ep_test_rewards=UtilsDataFrame.compress_decompress_list(ep_test_rewards_compressed[0],compress=False)
                ep_test_len=UtilsDataFrame.compress_decompress_list(ep_test_len_compressed[0],compress=False)

                current_mean_reward.append(validation_ep_selection(ep_test_rewards,n_test_ep))
                
                for_cumulative_n_test_timesteps=validation_ep_selection(ep_test_len,n_test_ep,test_type='len')
                cumulative_n_test_timesteps=[cumulative_n_test_timesteps[i]+for_cumulative_n_test_timesteps[i] for i in range(100)]

            else:
                current_mean_reward.append([-np.Inf]*100)
            current_n_test_timesteps.append(cumulative_n_test_timesteps)

        current_mean_reward=np.array(current_mean_reward).T
        current_n_test_timesteps=np.array(current_n_test_timesteps).T
        total_timesteps=[ i+np.array(df_train['n_train_timesteps']) for i in current_n_test_timesteps]
        min_total_timesteps=max([min(i) for i in total_timesteps])

        y=[]
        x_plot=[]
        for timesteps in x:
            if min_total_timesteps<timesteps:
                indx_max=[list(i<=timesteps).index(False) for i in total_timesteps]
                list_argmax=[np.argmax(current_mean_reward[i][:indx_max[i]])for i in range(100)]
                y_mean,y_q05,y_q95=UtilsFigure.bootstrap_mean_and_confidence_interval([df_test['mean_reward'][i] for i in list_argmax])
                y.append([y_mean,y_q05,y_q95])
                x_plot.append(timesteps)

        ax.fill_between(x_plot,np.array(y)[:,1],np.array(y)[:,2], alpha=.2, linewidth=0)

        if type(label) is int:
            label=['Every '+str(label)+' policies' if label!=1 else 'Every '+str(label)+' policy']
        else:
            label= 'Change in the best\n``Trajec. mean"'
        plt.plot(x_plot, np.array(y)[:,0], linewidth=1,label=label)

    ax.legend(title="",fontsize=9,handlelength=1,labelspacing=0.2)
    ax.set_xlabel("Total interaction steps",fontsize=13)
    ax.set_ylabel("$\widetilde{J}$ of the policy\nselected as the best",fontsize=13)
    ax.set_title('Evaluating policies with a frequency (legend)\n in additional 5 episodes',fontsize=14)
    plt.yticks(rotation=90)

    if len(list_n_test_ep)>1:
        ax.set_title('Learning-curves with\n\n Same validation freq: '+str(eval_freq)+' (policies)\nDifferent validation acc (test episodes, see legend)',fontsize=10)
        plt.savefig(output_path+'/DIFFacc_SAMEfreq'+str(eval_freq)+'_'+str(seed)+'.pdf')
    else:
        if eval_freq_with_train:

            plt.savefig(output_path+'/DIFFfreq_withtrain_SAMEacc'+str(n_test_ep)+'_'+str(seed)+'.pdf')

        else:
            plt.savefig(output_path+'/DIFFfreq_SAMEacc'+str(n_test_ep)+'_'+str(seed)+'.pdf')

    plt.show()
    plt.close()

def learning_curves_train_reward(x,env_name,seed):

    # Para guardar los resultados
    output_path='results/MAEB/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    fig=plt.figure(figsize=[5,3])
    plt.subplots_adjust(left=0.18,bottom=0.2,right=0.94,top=0.82,wspace=0.39,hspace=0.2)
    plt.rc('font', family='serif')
    plt.rc('text', usetex=True)
    plt.rcParams['text.latex.preamble'] = r'\boldmath'

    #----------------------------------------------------------------------------------------------
    # GRAFICA 1: learning-curves
    #----------------------------------------------------------------------------------------------
    ax=plt.subplot(111)
    ax.grid(True, which='both',linestyle='--', linewidth=0.8,alpha=0.2)
        
    train_reward_matrix = pickle.load(open('results/KnowingSingleProcess/'+str(env_name)+'/extracted_data/window_train_rewards'+str(seed)+'.pkl', 'rb'))

    df_test=pd.read_parquet('results/EnvironmentProcesses/'+str(env_name)+'/df_test_'+str(env_name)+'_seed'+str(seed)+'.parquet')
    df_train=pd.read_parquet('results/EnvironmentProcesses/'+str(env_name)+'/df_train_'+str(env_name)+'_seed'+str(seed)+'.parquet')
    list_window_sizes=[100,50,20,10,5,1]

    y_matrix=[]
    labels=[]
    colors=[]
    default_colors=list(mcolors.TABLEAU_COLORS.keys())

    # Dibujar curva sin test durante el proceso (ultima politica observada)
    y=[]
    for timesteps in x:
            last_policy=df_train[df_train['n_train_timesteps']<=timesteps]['n_policy'].max()+1
            y.append(float(df_test[df_test['n_policy']==last_policy]['mean_reward']))
    plt.plot(x, y, linewidth=1,color='black',label='None (last policy)')
    y_matrix.append(y)
    labels.append('None (last policy)')
    colors.append('black')
 
    # Resto de curvas usando window
    for i in [0,1,5]:
        y=[]
        for train_timesteps in x:
            last_policy=df_train[df_train['n_train_timesteps']<=train_timesteps]['n_policy'].max()
            best_policy=train_reward_matrix[i].index(max(train_reward_matrix[i][:last_policy]))
            y.append(float(df_test[df_test['n_policy']==best_policy+1]['mean_reward']))

        plt.plot(x, y, linewidth=1,label=str(list_window_sizes[i])+' previous episodes')
        y_matrix.append(y)
        labels.append(str(list_window_sizes[i])+'previous episodes')
        colors.append(default_colors[i+2])

    ax.legend(title="",fontsize=9,handlelength=1,labelspacing=0.2)
    ax.set_xlabel("Total interaction steps",fontsize=13)
    ax.set_ylabel("$\widetilde{J}$ of the policy\nselected as the best",fontsize=13)
    ax.set_title('Evaluating all policies with metrics (legend)\n from learning trajectories',fontsize=14)
    plt.yticks(rotation=90)
    plt.savefig(output_path+'/BestPolicyAdditionalInteraction'+str(seed)+'.pdf')
    plt.show()
    plt.close()
   

# Degradation
degradation(list(range(10000, 3200001, 32100)),'Ant',1,'Significant degradation')
degradation(list(range(10000, 3200001, 32100)),'Ant',3,'Catastrophic forgetting')
degradation(list(range(10000, 3200001, 32100)),'Ant',6,'Low degradation')

# Best policy using trajectories
metrics_train_changes=pickle.load(open('results/KnowingSingleProcess/Ant/extracted_data/metric_changes_norm1.pkl', 'rb'))
learning_curves_test_reward(list(range(10000, 3200001, 32100)),[5],'Ant',1,[1,300,700,['Defined by max change in "Trajec mean"',metrics_train_changes[1]]])


# Best policy using extra interaction
learning_curves_train_reward(list(range(10000, 3200001, 32100)),'Ant',1)
