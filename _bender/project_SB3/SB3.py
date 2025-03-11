'''
Script para ejecutar job con Stable Baselines en bender

NOTE: Para que el directorio en donde se guarde todo se identifique bien, definir 
(outputs es la carpeta que hay que definir en bender por cada proyecto):
os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

NOTE: en el cluster ajustar el minimo numero de parametros necesario en learn_process, para que el resto se definan por defecto. 

Hay que definir: 
- method
- env
- seed
- total_timesteps
- experiment_name
- library_dir
- n_eval_ep

Valores concretos:
- device='auto' # Para que detecte automaticamente lo disponible
- callback=True # Para validar durante el proceso
- eval_freq=2048 # (=n_steps_per_env*n_workers) es para que valide todas las politicas
- deterministic_eval=True # Para que la validacion sea pareada

NOTE: caracteristicas en el sbatch
Despues de un analisis he visto que para n_eval_ep=500, 32G, 16CPU, 1GPU son suficientes. 
Parece que esta libreria no esta diseñada para aprovechar el uso de GPU.

TODO: queda por probar si usar mi eval_policy para validar en paralelo es mas rapido que la validacion hecha por el Callback implementado.
'''

import numpy as np
import torch as th
import pandas as pd
from os.path import join
import os
import gymnasium as gym
from gymnasium import spaces
from typing import TypeVar, Any, Callable, Optional, Union, Tuple
from joblib import Parallel, delayed
import warnings
import json
import bz2
import base64
import numpy as np
import os
import time
import csv


from stable_baselines3.ppo import MlpPolicy, PPO
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv,VecEnv, sync_envs_normalization, VecMonitor, is_vecenv_wrapped
from stable_baselines3.common.type_aliases import  MaybeCallback
from stable_baselines3.common.callbacks import EvalCallback
SelfOnPolicyAlgorithm = TypeVar("SelfOnPolicyAlgorithm", bound="OnPolicyAlgorithm")

class MyTimer:
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.start_t = time.time()
        self.pause_t=0

    def pause(self):
        self.pause_start = time.time()
        self.paused=True

    def resume(self):
        if self.paused:
            self.pause_t += time.time() - self.pause_start
            self.paused = False

    def get_time(self):
        return time.time() - self.start_t - self.pause_t


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

class ModifiedFunctions:
    def learn(
        self: SelfOnPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfOnPolicyAlgorithm:
        
        ##############################MODIFICACION
        global process_dir, n_policy, total_time_seconds
        n_policy=0
        total_time_seconds=MyTimer()
        total_time_seconds.reset()

        total_time_seconds.pause()
        # Guardar politica inicial
        self.save(process_dir+'/policy'+str(n_policy)+'.zip')

        # Crear bases de datos donde ire escribiendo los datos por iteracion
        df_traj_csv=pd.DataFrame(columns=['n_policy','n_timesteps','time_seconds','traj_rewards','traj_ep_end','traj_inits'])
        df_traj_csv.to_csv(join(process_dir, "df_traj.csv"), index=False)
        df_val_csv=pd.DataFrame(columns=['n_policy','ep_inits','ep_rewards','ep_lens','n_val_ep','elapsed_val_time'])
        df_val_csv.to_csv(join(process_dir, "df_val.csv"), index=False)
        total_time_seconds.resume()
        #################################

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

            self.train()

            ###############################MODIFICACION
            total_time_seconds.pause()
            # Guardar politicas
            self.save(process_dir+'/policy'+str(n_policy)+'.zip')
            n_policy+=1
            total_time_seconds.resume()
            ######################################

        callback.on_training_end()

        return self
    
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
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
        global total_time_seconds, process_dir

        total_time_seconds.pause()
        global n_policy,df_traj
        policy_traj_rewards=[[] for _ in range(env.num_envs)]
        policy_traj_ep_end=[[] for _ in range(env.num_envs)]
        policy_traj_ep_inits=[[] for _ in range(env.num_envs)]
        total_time_seconds.resume()
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
            total_time_seconds.pause()#MODIFICACION: cuando no de define un callback esto no trada nada, pero cuando si se define esto tarda porque se hace validacion
            if not callback.on_step():
                return False
            total_time_seconds.resume()#MODIFICACION

            self._update_info_buffer(infos, dones)
            n_steps += 1

            ###############################MODIFICACION
            total_time_seconds.pause()
            for i in range(env.num_envs):
                policy_traj_rewards[i].append(float(rewards[i]))
                policy_traj_ep_end[i].append(True if dones[i] else False)
                if policy_traj_ep_end[i][-1]:
                    policy_traj_ep_inits[i].append(compress_decompress_list(new_obs[i].tolist()))
            total_time_seconds.resume()
            ################################

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstrapping with value function
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
        total_time_seconds.pause()
        with open(join(process_dir, "df_traj.csv"), 'a',newline='') as df_traj_csv:
            writer = csv.writer(df_traj_csv)
            writer.writerow([n_policy,self.num_timesteps,total_time_seconds.get_time(),compress_decompress_list(policy_traj_rewards),compress_decompress_list(policy_traj_ep_end),compress_decompress_list(policy_traj_ep_inits)])

        total_time_seconds.resume()
        ##########################

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True
    
    def _on_step(self) -> bool:

        #####################################
        global all_initial_states, eval_env_name, make_eval_deterministic, n_policy, make_vec_env_type

        def my_evaluate_policy(
            model: "type_aliases.PolicyPredictor",
            env: Union[gym.Env, VecEnv],
            n_eval_episodes: int = 10,
            deterministic: bool = True,
            render: bool = False,
            callback: Optional[Callable[[dict[str, Any], dict[str, Any]], None]] = None,
            reward_threshold: Optional[float] = None,
            return_episode_rewards: bool = False,
            warn: bool = True,
        ) -> Union[tuple[float, float], tuple[list[float], list[int]]]:
            """
            Runs policy for ``n_eval_episodes`` episodes and returns average reward.
            If a vector env is passed in, this divides the episodes to evaluate onto the
            different elements of the vector env. This static division of work is done to
            remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
            details and discussion.

            .. note::
                If environment has not been wrapped with ``Monitor`` wrapper, reward and
                episode lengths are counted as it appears with ``env.step`` calls. If
                the environment contains wrappers that modify rewards or episode lengths
                (e.g. reward scaling, early episode reset), these will affect the evaluation
                results as well. You can avoid this by wrapping environment with ``Monitor``
                wrapper before anything else.

            :param model: The RL agent you want to evaluate. This can be any object
                that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
                or policy (``BasePolicy``).
            :param env: The gym environment or ``VecEnv`` environment.
            :param n_eval_episodes: Number of episode to evaluate the agent
            :param deterministic: Whether to use deterministic or stochastic actions
            :param render: Whether to render the environment or not
            :param callback: callback function to do additional checks,
                called after each step. Gets locals() and globals() passed as parameters.
            :param reward_threshold: Minimum expected reward per episode,
                this will raise an error if the performance is not met
            :param return_episode_rewards: If True, a list of rewards and episode lengths
                per episode will be returned instead of the mean.
            :param warn: If True (default), warns user about lack of a Monitor wrapper in the
                evaluation environment.
            :return: Mean reward per episode, std of reward per episode.
                Returns ([float], [int]) when ``return_episode_rewards`` is True, first
                list containing per-episode rewards and second containing per-episode lengths
                (in number of steps).
            """
            is_monitor_wrapped = False
            # Avoid circular import
            from stable_baselines3.common.monitor import Monitor

            if make_eval_deterministic:# MODIFICADO
                if env.num_envs==1:
                    env=gym.make(eval_env_name)
                else:
                    if make_vec_env_type=='sequential':
                        env = make_vec_env(eval_env_name, n_envs=env.num_envs)
                    if make_vec_env_type=='parallel':
                        env = make_vec_env(eval_env_name, n_envs=env.num_envs,vec_env_cls=SubprocVecEnv)
                            

            if not isinstance(env, VecEnv):
                print('ENTRE AQUI')
                env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

            if make_eval_deterministic:
                env.seed(0)# MODIFICADO

            is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

            if not is_monitor_wrapped and warn:
                warnings.warn(
                    "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
                    "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
                    "Consider wrapping environment first with ``Monitor`` wrapper.",
                    UserWarning,
                )

            n_envs = env.num_envs
            episode_rewards = []
            episode_lengths = []
            episode_inits=[]#MODIFICADO
            start_val_time=time.time()#MODIFICADO
            num_episodes=[]#MODIFICADO
            times_per_episode=[]#MODIFICADO
            episode_inits_per_env=[[] for i in range(env.num_envs)]#MODIFICADO



            episode_counts = np.zeros(n_envs, dtype="int")
            # Divides episodes among different sub environments in the vector as evenly as possible
            episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

            current_rewards = np.zeros(n_envs)
            current_lengths = np.zeros(n_envs, dtype="int")
            observations = env.reset()

            for i in range(env.num_envs):#MODIFICADO
                episode_inits_per_env[i].append(observations[i])
 

            states = None
            episode_starts = np.ones((env.num_envs,), dtype=bool)
            while (episode_counts < episode_count_targets).any():
                actions, states = model.predict(
                    observations,  # type: ignore[arg-type]
                    state=states,
                    episode_start=episode_starts,
                    deterministic=deterministic,
                )
                new_observations, rewards, dones, infos = env.step(actions)
                current_rewards += rewards
                current_lengths += 1
                for i in range(n_envs):
                    if episode_counts[i] < episode_count_targets[i]:
                        # unpack values so that the callback can access the local variables
                        reward = rewards[i]
                        done = dones[i]
                        info = infos[i]
                        episode_starts[i] = done

                        if callback is not None:
                            callback(locals(), globals())

                        if dones[i]:
                            if is_monitor_wrapped:
                                # Atari wrapper can send a "done" signal when
                                # the agent loses a life, but it does not correspond
                                # to the true end of episode
                                if "episode" in info.keys():
                                    # Do not trust "done" with episode endings.
                                    # Monitor wrapper includes "episode" key in info if environment
                                    # has been wrapped with it. Use those rewards instead.
                                    episode_rewards.append(info["episode"]["r"])
                                    episode_lengths.append(info["episode"]["l"])
                                    # Only increment at the real end of an episode
                                    episode_counts[i] += 1
                                    num_episodes.append(sum(episode_counts))#MODIFICADO*
                                    times_per_episode.append(time.time()-start_val_time)#MODIFICADO*
                            else:
                                episode_rewards.append(current_rewards[i])
                                episode_lengths.append(current_lengths[i])
                                episode_counts[i] += 1
                                num_episodes.append(sum(episode_counts))#MODIFICADO
                                times_per_episode.append(time.time()-start_val_time)#MODIFICADO
                            current_rewards[i] = 0
                            current_lengths[i] = 0

                            episode_inits.append(compress_decompress_list(episode_inits_per_env[i][-1].tolist()))#MODIFICADO
                            episode_inits_per_env[i].append(new_observations[i])#MODIFICADO

                observations = new_observations

                if render:
                    env.render()

            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            if reward_threshold is not None:
                assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
            if return_episode_rewards:
                return episode_rewards, episode_lengths, episode_inits, num_episodes, times_per_episode # MODIFICADO
            return mean_reward, std_reward

        ###################################
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths, episode_inits, num_episodes, times_per_episode = my_evaluate_policy( #MODIFICADO
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            #######################################MODIFICADO
            # Escribir datos de validacion de la nueva politica
            with open(join(process_dir, "df_val.csv"), 'a',newline='') as df_val_csv:
                writer = csv.writer(df_val_csv)
                writer.writerow([n_policy,compress_decompress_list(episode_inits),compress_decompress_list(episode_rewards),compress_decompress_list([int(i) for i in episode_lengths]),compress_decompress_list([int(i) for i in num_episodes]),compress_decompress_list(times_per_episode)])
            #######################################

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)


                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)


                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,  # type: ignore[arg-type]
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = float(mean_reward)
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training
    
class Options:
    def learn_process(method,env_name,seed,total_timesteps,experiment_name,library_dir, # Parametros que determinan el proceso
                      n_steps_per_env=2048,n_workers=1, # Parametros que determinan la interaccion (aqui siempre n_envs_per_worker=1)
                      n_epoch=10,batch_size=64, # Parametros que determinan la actualizacion de politica
                      device='auto', vec_env_type='sequential', # Parametros que determinan el tipo de ejecucion (cpu,gpu)
                      callback=None, n_eval_ep=5, eval_freq=10000, n_eval_envs=1, deterministic_eval=False,stats_window_size=100, # tecnicas de rastreo
                      ):# Añadidas como predefinidas las variables/parametros que especifican la interaccion y la actualizacion de politica
        
        # Variables globales
        global df_traj, process_dir, all_initial_states, eval_env_name, make_eval_deterministic, make_vec_env_type
        df_traj=[]
        process_dir=library_dir+'/'+experiment_name+'/process_info'
        all_initial_states=[]
        eval_env_name=env_name
        make_eval_deterministic=deterministic_eval
        make_vec_env_type=vec_env_type

        # Crear nuevos directorios.
        os.makedirs(process_dir)

        # Modificar funciones de librerias existentes
        OnPolicyAlgorithm.learn=ModifiedFunctions.learn
        OnPolicyAlgorithm.collect_rollouts=ModifiedFunctions.collect_rollouts

        if callback==True:
            EvalCallback._on_step= ModifiedFunctions._on_step

        # Iniciar proceso de aprendizaje fijando el metodo, el env y la semilla.
        if n_workers==1:
            env=gym.make(env_name)
        else:
            if vec_env_type=='sequential':# Por defecto es DummyVecEnv que se ejecuta en secuencial
                env = make_vec_env(env_name, n_envs=n_workers)
            if vec_env_type=='parallel':
                env = make_vec_env(env_name, n_envs=n_workers,vec_env_cls=SubprocVecEnv)

        if n_eval_envs==1:
            eval_env=gym.make(env_name)
        else:
            if vec_env_type=='sequential':
                eval_env = make_vec_env(env_name, n_envs=n_eval_envs)
            if vec_env_type=='parallel':
                eval_env = make_vec_env(env_name, n_envs=n_eval_envs,vec_env_cls=SubprocVecEnv)

        model = PPO(MlpPolicy,
                    env, seed=seed,
                    n_steps=n_steps_per_env,batch_size=batch_size,n_epochs=n_epoch,
                    stats_window_size=stats_window_size,
                    verbose=0,device=device)# TODO: la primera linea modificarla para otros algoritmos
        model.set_random_seed(seed)

        if callback==True:
            callback=EvalCallback(eval_env,n_eval_episodes=n_eval_ep,eval_freq=eval_freq,
                                  log_path=library_dir+'/'+experiment_name,best_model_save_path=library_dir+'/'+experiment_name)

        model.learn(total_timesteps=total_timesteps,callback=callback)


        # Guardar el modelo output
        model.save(library_dir+'/'+experiment_name+'/policy_output.zip')

    def eval_policy(policy_id,env,seed,n_eval_ep,process_dir,
                    n_workers=1):
        '''
        TODO: esta por comprobar si las mediciones de tiempo que he añadido funcionan bien.

        Esta funcion puedo usarla como alternativa a _on_step que la usa por defecto el Callback. 
        Tendria que probar si va mas rapido, aunque deberia ir parecido a cuando defino vec_env_type='sequential' 
        que define el eval_env como SubprocVecEnv (considera ejecucion paralela) en lugar del DummyVecEnv (ejecucion secuencial) por defecto.

        La medicion de los tiempos esta pesnada para registrar el intervalo de tiempo [inicio, fin] de la ejecucion de cada episodio, y despues
        poder simular los tiempos que se consumen con validaciones de menos episodios.
        '''
        
        
        def eval_single_episode(args):
            env_name,policy,episode,global_init_time=args
            ep_start_time=time.time()
        
            env=gym.make(env_name)
            if not isinstance(env, VecEnv):
                env = DummyVecEnv([lambda: env])
            env.seed(0)

            obs=[env.reset() for _ in range(episode)][-1]# La lista de estados iniciales con interaccion coincide con los estados iniciales impares sin interaccion.

            episode_rewards = 0
            episode_len=0
            done = False # Parameter that indicates after each action if the episode continues (False) or is finished (True).

            with th.no_grad():
                while not done:
                    action, _states = policy.predict(obs, deterministic=True) # The action to be taken with the model is predicted.       
                    obs, reward, done, info = env.step(action) # Action is applied in the environment.
                    episode_rewards+=reward # The reward is saved.
                    episode_len+=1

            return episode_rewards, episode_len, obs, [ep_start_time-global_init_time,time.time()-global_init_time]

        def parallel_eval(policy,env_name,n_eval_ep,n_workers):
            global_init_time = time.time()
            # Set up the parallel processing pool
            results=Parallel(n_jobs=n_workers, backend="loky")(
                    delayed(eval_single_episode)([env_name,policy,episode,global_init_time]) for episode in range(1,n_eval_ep+1))
                
            # Split the results into rewards and episode lengths
            all_episode_reward, all_episode_len, all_init_state, all_time_intervals= zip(*results)

            #return np.mean(all_episode_reward), np.std(all_episode_reward), [float(i) for i in all_episode_reward], [int(i)for i in all_episode_len], all_init_state
            return [float(i) for i in all_episode_reward], all_episode_len, np.array(all_init_state)
        # Cragar la politica.
        policy=PPO.load(process_dir+'/'+str(policy_id)) #TODO: tras llamar a process_learn se deberia crear un fichero con la info de la conf para saber que algo hemos usado

        # Evaluar la politica.
        eval_metrics=parallel_eval(policy,env,n_eval_ep,n_workers)

        # Guardar datos de evaluacion.
        #print('ep_mean: '+str(ep_mean)+';  ep_std: '+str(ep_std)) # TODO: esto se puede hacer mas sofisticado, con otras metricas, guardar en .csv.
        return eval_metrics
 
# Probando si funciona. 
method='PPO'
env='Ant-v4'
seed=1
total_timesteps=2048*3
library_dir='_bender/project_SB3/outputs'

# Options.learn_process(method,env,seed,total_timesteps,'execution1',library_dir,
#                       device='cpu',
#                       callback=True,n_eval_ep=2,eval_freq=2048,n_eval_envs=1,deterministic_eval=True)

# Options.learn_process(method,env,seed,total_timesteps,'execution2',library_dir,
#                       n_workers=2,
#                       device='cpu',
#                       callback=True,n_eval_ep=2,eval_freq=2048,n_eval_envs=2,deterministic_eval=True)

# if __name__ == "__main__": # Cuando vec_env_type='parallel' hay que ejecutarlo con esta linea porque habra ejecucion en paralelo
#     Options.learn_process(method,env,seed,total_timesteps,'execution3',library_dir,
#                         n_workers=2,vec_env_type='parallel',
#                         device='cpu',
#                         callback=True,n_eval_ep=2,eval_freq=2048,n_eval_envs=2,deterministic_eval=True)

# if __name__ == "__main__": # Cuando vec_env_type='parallel' hay que ejecutarlo con esta linea porque habra ejecucion en paralelo
#     Options.learn_process(method,env,seed,total_timesteps,'execution4',library_dir,
#                         n_workers=1,vec_env_type='parallel',
#                         device='cpu',
#                         callback=True,n_eval_ep=2,eval_freq=2048,n_eval_envs=2,deterministic_eval=True)

# Comprobar que los estados iniciales de validacion son los mismos
df_val=pd.read_csv('_bender/project_SB3/outputs/execution1/process_info/df_val.csv')
print([np.array(compress_decompress_list(i,compress=False)) for i in np.array(compress_decompress_list(df_val['ep_inits'][0],compress=False))])
print([np.array(compress_decompress_list(i,compress=False)) for i in np.array(compress_decompress_list(df_val['ep_inits'][1],compress=False))])

# Dimensiones de trajectorias 
df_val=pd.read_csv('_bender/project_SB3/outputs/execution3/process_info/df_traj.csv')
print(np.array(compress_decompress_list(df_val['traj_rewards'][0],compress=False)).shape)
print(np.array(compress_decompress_list(df_val['traj_ep_end'][0],compress=False)).shape)

print([np.count_nonzero(np.array(i)) for i in compress_decompress_list(df_val['traj_ep_end'][0],compress=False)])
print([len(i) for i in compress_decompress_list(df_val['traj_inits'][0],compress=False)])

# Comprobar que para env train (secuencial) y test (paralelo/vectorial) diferentes se guardan bien todos los datos
df_val=pd.read_csv('_bender/project_SB3/outputs/execution4/process_info/df_val.csv')

print(np.array(compress_decompress_list(df_val['ep_rewards'][1],compress=False)))
print(np.array(compress_decompress_list(df_val['n_val_ep'][1],compress=False)))
print(np.array(compress_decompress_list(df_val['elapsed_val_time'][1],compress=False)))
