import pandas as pd
import os

data_common_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
pack='pack_PPO_HalfCheetah' 
seeds=list(range(1,31))


df_test_all_seq_est = pd.DataFrame()
df_train_all_seq_est = pd.DataFrame()

for seed in seeds:  

    path_train=data_common_path+'/'+pack+'/'+pack+'_seed'+str(seed)+'_/process_info/df_traj_estimates.csv'
    path_test=data_common_path+'/'+pack+'/'+pack+'_seed'+str(seed)+'_/process_info/df_val_estimates.csv'

    df_train= pd.read_csv(path_train)
    df_test= pd.read_csv(path_test)

    # Seleccionar columnas de interes
    df_train = df_train[[c for c in df_train.columns if 'traj_ep' in c]].copy()
    df_test = df_test[[c for c in df_test.columns if 'val_ep' in c]].copy()

    # Renombrar
    df_train = df_train.rename(columns=lambda c: f"{c}_seed{seed}")
    df_test = df_test.rename(columns=lambda c: f"{c}_seed{seed}")

    # Concatenar horizontalmente
    df_train_all_seq_est = pd.concat([df_train_all_seq_est, df_train], axis=1)
    df_test_all_seq_est = pd.concat([df_test_all_seq_est, df_test], axis=1)

# Guardar bases de datos completas
path_train=data_common_path+'/'+pack+'/analysis_data/df_train_all_seq_est.csv'
path_test=data_common_path+'/'+pack+'/analysis_data/df_test_all_seq_est.csv'
df_train_all_seq_est.to_csv(path_train, index=False)
df_test_all_seq_est.to_csv(path_test, index=False)

