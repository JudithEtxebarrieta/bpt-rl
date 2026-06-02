import pandas as pd
import os
import numpy as np
import bz2
import base64
import json

def compress_decompress_list(my_list,compress=True):
    '''
    Sirve para comprimir o descomprimir listas almacenadas en las columnas de las bases de datos.
    '''
    if compress:
        # Convertir la lista a una cadena JSON compacta
        json_str = json.dumps(my_list)

        # Comprimir la cadena
        compressed_data = bz2.compress(json_str.encode('utf-8'))

        # Convertir a base64 para guardar como texto en la base de datos
        compressed_str = base64.b64encode(compressed_data).decode('utf-8')

        return compressed_str
    else:
        # Leer la cadena comprimida de la base de datos
        compressed_data = base64.b64decode(my_list.encode('utf-8'))

        # Descomprimir la cadena
        json_str = bz2.decompress(compressed_data).decode('utf-8')

        # Convertir la cadena JSON de vuelta a lista
        my_list = json.loads(json_str)

        return my_list
        

data_common_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
pack='pack_PPO_HalfCheetah' 
seeds=list(range(1,31))

path_save=data_common_path+'/'+pack+'/analysis_data/df_test_all_seq_cost_len.csv'
df_test_all_seq_est = pd.DataFrame()
df_train_all_seq_est = pd.DataFrame()

val_ep=[500,250,100,50,25,5]

df_test_all_seq_cost_len = pd.DataFrame()

for seed in seeds:  

    # columns test: n_policy,ep_inits,ep_rewards,ep_lens,n_val_ep,elapsed_val_time
    # columns train: n_policy,n_timesteps,time_seconds,traj_rewards,traj_ep_end,traj_inits,traj_advantages,traj_values,traj_returns,policy_loss,value_loss,entropy_loss,policy_gradient_loss,KL_div,explained_variance,log_std
    path_test=data_common_path+'/'+pack+'/'+pack+'_seed'+str(seed)+'_/process_info/df_val.csv'
    path_train=data_common_path+'/'+pack+'/'+pack+'_seed'+str(seed)+'_/process_info/df_traj.csv'

    df_test=pd.read_csv(path_test)
    df_test['ep_rewards']=[ compress_decompress_list(i,compress=False) for i in df_test['ep_rewards']]
    df_test['ep_lens']=[ compress_decompress_list(i,compress=False) for i in df_test['ep_lens']]
    df_train=pd.read_csv(path_train)

    df_test_all_seq_cost_len['ep_len_seed'+str(seed)]=[np.mean(i) for i in df_test['ep_lens']]

    for n_ep in val_ep:

        # Validation-to-learn cost ratios
        train_times=df_train['time_seconds'].tolist()
        train_iter_times= [0] + [train_times[i] - train_times[i - 1]  for i in range(1, len(train_times)) ]

        val_times=[]
        for i in range(df_test.shape[0]):
            df_test_elapsed_val_time=compress_decompress_list(df_test['elapsed_val_time'][i],compress=False)
            df_test_n_val_ep=compress_decompress_list(df_test['n_val_ep'][i],compress=False)
            val_time_until_truth=df_test_elapsed_val_time[df_test_n_val_ep.index(500)] 
            val_time=df_test_elapsed_val_time[df_test_n_val_ep.index(500+n_ep)] 
            val_times.append(val_time-val_time_until_truth)
        
        val_to_learn_rations=np.array(val_times)/np.array(train_iter_times)
        
        df_test_all_seq_cost_len[str(n_ep)+'_val_ep_seed'+str(seed)]=val_to_learn_rations
        df_test_all_seq_cost_len.to_csv(path_save, index=False)







