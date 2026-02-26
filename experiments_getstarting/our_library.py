
from stable_baselines3.ppo import MlpPolicy, MultiInputPolicy, PPO
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.utils import  obs_as_tensor
from stable_baselines3.common.type_aliases import  MaybeCallback 
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import RolloutBuffer
import numpy as np
import pandas as pd
from gymnasium import spaces
import torch as th
from torch.nn import functional as F
from typing import  TypeVar
from tqdm import tqdm
import gymnasium as gym
import json
import bz2
import base64
import os
from joblib import Parallel, delayed

class PolicyValidation:
    '''
    La ultima modificacion hecha en este fichero our_library.py es la construccion de esta clase, en donde he metido todas las funciones asociadas
    con la validacion de politicas. Antes solo consideraba la validacion secuencial, es decir la evaluacion de la politica en un conjunto de
    episodios, evaluando uno tras otro (funcion evaluate). Ahora he añadido las funciones necesarias para poder evaluar los episodios en paralelo.
    Por eso, ciertos scripts de experimentos puede que necesiten importar esta nueva clase para que el evaluate antiguo se reconozca.
    '''
    def evaluate_single_episode(args):

        env_name,policy,episode=args
    
        env=gym.make(env_name)
        if not isinstance(env, VecEnv):
            env = DummyVecEnv([lambda: env])
        env.seed(0)

        obs=[env.reset() for _ in range(2*(episode-1)+1)][-1]# La lista de estados iniciales con interaccion coincide con los estados iniciales impares sin interaccion.

        episode_rewards = 0
        episode_len=0
        done = False # Parameter that indicates after each action if the episode continues (False) or is finished (True).

        with th.no_grad():
            while not done:
                action, _states = policy.predict(obs, deterministic=True) # The action to be taken with the model is predicted.       
                obs, reward, done, info = env.step(action) # Action is applied in the environment.
                episode_rewards+=reward # The reward is saved.
                episode_len+=1

        return episode_rewards, episode_len


    def parallel_evaluate(policy,env_name,n_eval_episodes,n_processes):

        # Set up the parallel processing pool
        results=Parallel(n_jobs=n_processes, backend="loky")(
                delayed(PolicyValidation.evaluate_single_episode)([env_name,policy,episode]) for episode in tqdm(range(1,n_eval_episodes+1)))
            
        # Split the results into rewards and episode lengths
        all_episode_reward, all_episode_len = zip(*results)

        return np.mean(all_episode_reward), np.std(all_episode_reward), [float(i) for i in all_episode_reward], [int(i)for i in all_episode_len]



    def evaluate(policy,env,n_eval_episodes,seed=0):
        '''
        The current policy is evaluated using the episodes of the validation environment.

        Parameters
        ==========
        policy: Policy to be evaluated.
        env : Validation environment.
        n_eval_episodes (int): Number of episodes (evaluations) in which the model will be evaluated.
        seed (int): Seed of the validation environment (by default 0).

        Returns
        =======
        Average and standard deviation of the rewards obtained in the n_eval_episodes episodes.
        '''
        # To save the reward per episode.
        all_episode_reward=[]
        all_episode_len=[]

        # To ensure that the same episodes are used in each call to the function.
        if not isinstance(env, VecEnv):
            env = DummyVecEnv([lambda: env])
        env.seed(seed)
        obs=env.reset()
        
        with th.no_grad():# Para que los datos obtenidos durante la validacion no se usen a la hora de actualizar la politica posteriormente.

            for _ in range(n_eval_episodes):

                episode_rewards = 0
                episode_len=0
                done = False # Parameter that indicates after each action if the episode continues (False) or is finished (True).
                while not done:
                    action, _states = policy.predict(obs, deterministic=True) # The action to be taken with the model is predicted.         
                    obs, reward, done, info = env.step(action) # Action is applied in the environment.
                    episode_rewards+=reward # The reward is saved.
                    episode_len+=1

                # Save total episode reward.
                all_episode_reward.append(episode_rewards)
                all_episode_len.append(episode_len)
            
                # Reset the episode.
                obs = env.reset() 

        
        return np.mean(all_episode_reward), np.std(all_episode_reward), [float(i) for i in all_episode_reward], [int(i)for i in all_episode_len]

class UtilsFigure:
    def bootstrap_mean_and_confidence_interval(data,bootstrap_iterations=1000):
        '''
        The 95% confidence interval of a given data sample is calculated.

        Parameters
        ==========
        data (list): Data on which the range between percentiles will be calculated.
        bootstrap_iterations (int): Number of subsamples of data to be considered to calculate the percentiles of their means. 

        Return
        ======
        The mean of the original data together with the percentiles of the means obtained from the subsampling of the data. 
        '''
        mean_list=[]
        for i in range(bootstrap_iterations):
            sample = np.random.choice(data, len(data), replace=True) 
            mean_list.append(np.mean(sample))
        return np.mean(data),np.quantile(mean_list, 0.05),np.quantile(mean_list, 0.95)
    
class UtilsDataFrame:

    def join_csv_and_save_parquet(list_csv_name,read_path,save_path,joined_name):

        df=pd.read_csv(read_path+list_csv_name[0],index_col=0)
        os.remove(read_path+list_csv_name[0])

        for i in range(1,len(list_csv_name)):
            df_new=pd.read_csv(read_path+list_csv_name[i],index_col=0)
            df = pd.concat([df, df_new], ignore_index=False)
            os.remove(read_path+list_csv_name[i])

        df.to_parquet(save_path+joined_name+'.parquet', engine='pyarrow', compression='gzip',index=False)

    def compress_decompress_list(my_list,compress=True):
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

class FromStableBaselines3:
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        '''
        Modificada para guardar datos recolectados durante la interccion de las politicas con el entorno (train rewards).
        '''

        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        ########################MODIFICACION
        global n_policy,df_train
        policy_train_rewards=[]
        policy_train_ep_end=[]
        #########################


        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict

                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            
            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            ###############################MODIFICACION
            policy_train_rewards.append(float(rewards[0]))
            policy_train_ep_end.append(True if dones[0] else False)

            ################################


            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        #########################MODIFICACION
        df_train.append([self.seed,n_policy,self.num_timesteps,UtilsDataFrame.compress_decompress_list(policy_train_rewards),UtilsDataFrame.compress_decompress_list(policy_train_ep_end)])
        ##########################

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True
    
    def learn(
        self: TypeVar("SelfOnPolicyAlgorithm", bound="OnPolicyAlgorithm"),
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> TypeVar("SelfOnPolicyAlgorithm", bound="OnPolicyAlgorithm"):
        
        '''
        Modificada para guardar datos de validacion de politicas durante el entrenamiento.
        '''

        ##########################MODIFICACION
        global env_name,df_test,df_train,global_path,global_csv_name, global_partial_save,global_also_train,global_also_models,n_policy, n_eval_episodes, eval_env, test_timesteps 
        n_appended=0
        appended=0
        ##########################

        iteration = 0


        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        

        with tqdm(total=total_timesteps,desc='Learning process: ') as process_bar: #MODIFICACION

            while self.num_timesteps < total_timesteps:

                continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

                if not continue_training:
                    break

                iteration += 1
                self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

                # Display training infos
                if log_interval is not None and iteration % log_interval == 0:
                    assert self.ep_info_buffer is not None
                    self._dump_logs(iteration)

                ########################MODIFICACION
                n_policy+=1
                mean_reward, std_reward, ep_test_rewards,ep_test_len = PolicyValidation.evaluate(self.policy, eval_env, n_eval_episodes)
                test_timesteps+=sum(ep_test_len)
                df_test.append([self.seed,n_policy,test_timesteps,mean_reward,UtilsDataFrame.compress_decompress_list(ep_test_len),UtilsDataFrame.compress_decompress_list(ep_test_rewards)])
                appended+=1

                if global_partial_save is not False:
                    if appended%global_partial_save==0:
                        appended=0
                        n_appended+=1
                        df_test=pd.DataFrame(df_test,columns=['seed','n_policy','n_test_timesteps','mean_reward','ep_test_len','ep_test_rewards'])
                        df_test.to_csv(global_path+'df_test_'+str(n_appended)+'_'+global_csv_name, index=False)
                        df_test=[]
                        if global_also_train:
                            df_train=pd.DataFrame(df_train,columns=['seed','n_policy','n_train_timesteps','train_rewards','train_ep_end'])
                            df_train.to_csv(global_path+'df_train_'+str(n_appended)+'_'+global_csv_name, index=False)
                            df_train=[]
                    
                # Guardar tambien los modelos
                if global_also_models:
                    self.save(global_path+'policies_'+env_name+'_seed'+str(self.seed)+'/policy'+str(n_policy)+'.zip')

                ########################


                
                self.train()

                # Para saber por donde voy
                process_bar.update(self.n_steps)

        callback.on_training_end()


        return callback #MODIFICACION
        

    def resume_learn(
        self: TypeVar("SelfOnPolicyAlgorithm", bound="OnPolicyAlgorithm"),
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1
    ) -> TypeVar("SelfOnPolicyAlgorithm", bound="OnPolicyAlgorithm"):
        
        '''
        Version inspirada en la learn anterior para reanudar ejecuciones.
        '''
        
        ###############################MODIFICACION
        global df_test, n_policy, n_eval_episodes, eval_env,test_timesteps, global_path, global_csv_name,global_partial_save,global_also_train
        n_appended=0
        appended=0
        ##############################

        
        iteration=0
               
        assert self.env is not None
        
      
        self.num_timesteps=0
        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                self._dump_logs(iteration)

            ########################MODIFICACION
            n_policy+=1
            mean_reward, std_reward, ep_test_rewards,ep_test_len = PolicyValidation.evaluate(self.policy, eval_env, n_eval_episodes)
            test_timesteps+=sum(ep_test_len)
            df_test.append([self.seed,n_policy,test_timesteps,mean_reward,UtilsDataFrame.compress_decompress_list(ep_test_len),UtilsDataFrame.compress_decompress_list(ep_test_rewards)])

            appended+=1

            if global_partial_save is not False:
                if appended%global_partial_save==0:
                    appended=0
                    n_appended+=1
                    df_test=pd.DataFrame(df_test,columns=['seed','n_policy','n_test_timesteps','mean_reward','ep_test_len','ep_test_rewards'])
                    df_test.to_csv(global_path+'df_test_'+str(n_appended)+'_'+global_csv_name, index=False)
                    df_test=[]
                    if global_also_train:
                        df_train=pd.DataFrame(df_train,columns=['seed','n_policy','n_train_timesteps','train_rewards','train_ep_end'])
                        df_train.to_csv(global_path+'df_train_'+str(n_appended)+'_'+global_csv_name, index=False)
                        df_train=[]

            ########################

            self.train()

        callback.on_training_end()
        
        return self,callback

class PPOLearner:
       

    def start_learn_process(env,seed,total_timesteps,n_test_episodes, path, csv_name,also_train=False,also_models=False,verbose=0,partial_save=False):

        global env_name,df_train,df_test, n_policy, eval_env, n_eval_episodes, test_timesteps,global_path,global_csv_name,global_partial_save,global_also_train,global_also_models
        env_name=env.spec.name
        global_path=path
        global_csv_name=csv_name
        global_partial_save=partial_save
        global_also_train=also_train
        global_also_models=also_models
        n_policy=0
        test_timesteps=0
        df_train=[]
        df_test=[]

        # Crear nuevos directorios.
        if global_also_models:
            os.makedirs(global_path+'policies_'+env_name+'_seed'+str(seed))

        # Modificar funciones de librerias existentes
        OnPolicyAlgorithm.learn=FromStableBaselines3.learn
        OnPolicyAlgorithm.collect_rollouts=FromStableBaselines3.collect_rollouts

        # Initialize PPO algorithm with policy, environment and the random seed
        if env.spec.id=="FetchReachDense-v2":
            
            policy=MultiInputPolicy
        else:
            policy=MlpPolicy
        
        model = PPO(policy, env, seed=seed,verbose=verbose)
        model.set_random_seed(seed)
        


        # Define evaluation environment
        eval_env=gym.make(env.spec.id)
        n_eval_episodes=n_test_episodes

        # Save data during training
        callback=model.learn(total_timesteps=total_timesteps)

        if also_train:
            df_train=pd.DataFrame(df_train,columns=['seed','n_policy','n_train_timesteps','train_rewards','train_ep_end'])
            df_train.to_csv(path+'df_train_'+csv_name, index=False)

        df_test=pd.DataFrame(df_test,columns=['seed','n_policy','n_test_timesteps','mean_reward','ep_test_len','ep_test_rewards'])
        df_test.to_csv(path+'df_test_'+csv_name, index=False)

        # Get random process states
        random_seed_state=[np.random.get_state(),th.random.get_rng_state()]
        
        return model, callback, n_policy, test_timesteps, random_seed_state
     


    def resume_learn_process_online(process,total_timesteps, path,csv_name, seed=None, n_test_episodes=None,also_train=False,partial_save=False):

        global df_train,df_test, n_eval_episodes, n_policy, test_timesteps,global_path,global_csv_name,global_partial_save,global_also_train
        global_path=path
        global_csv_name=csv_name
        global_partial_save=partial_save
        global_also_train=also_train
        df_train=[]
        df_test=[]

        # Modificar funciones de librerias existentes
        OnPolicyAlgorithm.collect_rollouts=FromStableBaselines3.collect_rollouts

        # Recuperar proceso de aprendizaje
        model,callback,n_policy,test_timesteps,random_seed_state=process
        if seed!= None:
            model.set_random_seed(seed)
        else:
            np.random.set_state(random_seed_state[0])
            th.random.set_rng_state(random_seed_state[1])

        if n_test_episodes!=None:
            n_eval_episodes=n_test_episodes

        # Reanudar aprendizaje
        model,callback=FromStableBaselines3.resume_learn(self=model,total_timesteps=total_timesteps,callback=callback)

        if also_train:
            df_train=pd.DataFrame(df_train,columns=['seed','n_policy','n_train_timesteps','train_rewards','train_ep_end'])
            df_train.to_csv(path+'df_train_'+csv_name, index=False)


        df_test=pd.DataFrame(df_test,columns=['seed','n_policy','n_test_timesteps','mean_reward','ep_test_len','ep_test_rewards'])
        df_test.to_csv(path+'df_test_'+csv_name, index=False)


        # Get random process states
        random_seed_state=[np.random.get_state(),th.random.get_rng_state()]


        return model, callback, n_policy,test_timesteps, random_seed_state
    

    def load_policy(path):
        loaded_policy=PPO.load(path)
        return loaded_policy

    


    


