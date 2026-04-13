'''
Script para generar procesos con Stable Baselines en bender.

NOTE: Para que el directorio en donde se guarde todo se identifique bien, definir 
(outputs es la carpeta que hay que definir en bender por cada proyecto):
os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

NOTE: Cuando vec_env_type='parallel' , ya sea porque n_workers>1 o n_eval_envs>1, el OnPolicy_learn_process/OffPolicy_learn_process 
hay que ejecutarlo dentro de esta linea porque habra ejecucion en paralelo: "if __name__ == "__main__":"

NOTE: en el cluster ajustar el minimo numero de parametros necesario en OnPolicy_learn_process/OffPolicy_learn_process, para que el resto se definan por defecto. 

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
- eval_freq=2048 # (=n_steps_per_env*n_workers) es para que valide todas las politicas NOTE: esto en PPO (OnPolicy), en SAC (OffPolicy) 1024 para que valide el doble que PPO (aunque se estara saltando la validacion de politicas) 
- deterministic_eval=True # Para que la validacion sea pareada

NOTE: caracteristicas en el sbatch para metodos OnPolicy_learn_process/OffPolicy_learn_process
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
from torch.nn import functional as F
import torch.nn as nn


from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import obs_as_tensor, explained_variance, should_collect_more_steps, polyak_update
from stable_baselines3.common.buffers import RolloutBuffer, ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv,VecEnv, sync_envs_normalization, VecMonitor, is_vecenv_wrapped
from stable_baselines3.common.type_aliases import  MaybeCallback, TrainFreq, RolloutReturn, TrainFrequencyUnit
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import ActionNoise
from rl_zoo3.wrappers import FrameSkip, YAMLCompatResizeObservation
from gymnasium.wrappers import GrayscaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize

SelfOffPolicyAlgorithm = TypeVar("SelfOffPolicyAlgorithm", bound="OffPolicyAlgorithm")
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

class ModifiedFunctions_Common:
    # Funcion que define la interaccion de validacion: se modifica para guardar los datos de validacion
    def _on_step(self) -> bool:

        #####################################
        global all_initial_states, eval_env_name, make_eval_deterministic,n_envs_for_truth_eval, n_policy, make_vec_env_type, make_eval_for_test

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

                if make_eval_for_test:

                    if env.num_envs==1:
                        env=gym.make(eval_env_name)
                    else:
                        if make_vec_env_type=='sequential':
                            env = make_vec_env(eval_env_name, n_envs=env.num_envs)
                        if make_vec_env_type=='parallel':
                            env = make_vec_env(eval_env_name, n_envs=env.num_envs,vec_env_cls=SubprocVecEnv)

                if not make_eval_for_test:

                    if env.num_envs==1:
                        env=gym.make(eval_env_name)
                    else:
                        if make_vec_env_type=='sequential':
                            env = make_vec_env(eval_env_name, n_envs=n_envs_for_truth_eval)
                        if make_vec_env_type=='parallel':
                            env = make_vec_env(eval_env_name, n_envs=n_envs_for_truth_eval,vec_env_cls=SubprocVecEnv)



            if not isinstance(env, VecEnv):
                print('ENTRE AQUI')
                env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

            if make_eval_deterministic:# MODIFICADO: para que sea determinista y considerar diferentes estados iniciales para test y para truth
                if make_eval_for_test: 
                    env.seed(99999)
                else:
                    env.seed(0)

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

            # MODIFICACION: Primero validacion para datos truth
            make_eval_for_test=False
            episode_rewards_truth, episode_lengths_truth, episode_inits_truth, num_episodes_truth, times_per_episode_truth = my_evaluate_policy( #MODIFICADO
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            #MODIFICACION: despues validacion para simulacion test
            make_eval_for_test=True
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
                #writer.writerow([n_policy,compress_decompress_list(episode_inits_truth+episode_inits),compress_decompress_list(episode_rewards_truth+episode_rewards),compress_decompress_list([int(i) for i in episode_lengths_truth+episode_lengths]),compress_decompress_list([int(i) for i in num_episodes_truth+list(np.array(num_episodes)+num_episodes_truth[-1])]),compress_decompress_list([0]*len(times_per_episode_truth)+times_per_episode)])
                writer.writerow([n_policy,None,compress_decompress_list(episode_rewards_truth+episode_rewards),compress_decompress_list([int(i) for i in episode_lengths_truth+episode_lengths]),compress_decompress_list([int(i) for i in num_episodes_truth+list(np.array(num_episodes)+num_episodes_truth[-1])]),compress_decompress_list([0]*len(times_per_episode_truth)+times_per_episode)])

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
    
class ModifiedFunctions_OnPolicy:
    # Funcion principal de aprendizaje: se modifica para inicializar bases de datos que almacenaran los datos para la simulacion y guardar las politicas.
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
        global process_dir, n_policy, total_time_seconds, make_policy_saving
        n_policy=0
        total_time_seconds=MyTimer()
        total_time_seconds.reset()

        total_time_seconds.pause()
        # Guardar politica inicial
        if make_policy_saving:
            self.save(process_dir+'/policy'+str(n_policy)+'.zip')

        # Crear bases de datos donde ire escribiendo los datos por iteracion
        df_traj_csv=pd.DataFrame(columns=['n_policy','n_timesteps','time_seconds','traj_rewards','traj_ep_end','traj_inits',
                                          'traj_advantages','traj_values','traj_returns',
                                          'policy_loss','value_loss','entropy_loss','policy_gradient_loss',
                                          'KL_div','explained_variance','log_std'])
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
            if make_policy_saving:
                self.save(process_dir+'/policy'+str(n_policy)+'.zip')
            n_policy+=1
            total_time_seconds.resume()
            ######################################

        callback.on_training_end()

        return self
    
    # Funcion para la interaccion de train: se modifica para guardar los datos de interaccion
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
            total_time_seconds.pause()#MODIFICACION: cuando no se define un callback esto no tarda nada, pero cuando si se define esto tarda porque se hace validacion
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

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        #########################MODIFICACION
        total_time_seconds.pause()
        with open(join(process_dir, "df_traj.csv"), 'a',newline='') as df_traj_csv:
            writer = csv.writer(df_traj_csv)
            # writer.writerow([n_policy,self.num_timesteps,total_time_seconds.get_time(),compress_decompress_list(policy_traj_rewards),compress_decompress_list(policy_traj_ep_end),compress_decompress_list(policy_traj_ep_inits),
            #                  compress_decompress_list(rollout_buffer.advantages.tolist()),compress_decompress_list(rollout_buffer.values.tolist()),compress_decompress_list(rollout_buffer.returns.tolist()),None,None,None,None])
            writer.writerow([n_policy,self.num_timesteps,total_time_seconds.get_time(),compress_decompress_list(policy_traj_rewards),compress_decompress_list(policy_traj_ep_end),None,
                    compress_decompress_list(rollout_buffer.advantages.tolist()),compress_decompress_list(rollout_buffer.values.tolist()),compress_decompress_list(rollout_buffer.returns.tolist()),None,None,None,None])

            # writer.writerow([n_policy,self.num_timesteps,total_time_seconds.get_time(),None,None,None,
            #         None,None,None,None,None,None,None])

        total_time_seconds.resume()
        ##########################

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    # Funcion para la actualizacion de politica: se modifica para guardar los losses de las politicas usados para su actualizacion
    def PPO_train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

        ####################################MODIFICACION
        # Terminar de completar datos de iteracion train guardando: 
        # - policy_loss: el loss usado para actualizar la politica
        # - entropy_loss: mide la incertidumbre/estocasticidad de la politica
        # - KL_div: mide lo diferentes que son la politica actual/actualizada y la anterior 
        global total_time_seconds
        total_time_seconds.pause()
        df_traj_csv = pd.read_csv(join(process_dir, "df_traj.csv"))
        
        df_traj_csv.iloc[-1, df_traj_csv.columns.get_loc('policy_loss')] = loss.item()
        df_traj_csv.iloc[-1, df_traj_csv.columns.get_loc('value_loss')] = np.mean(value_losses)
        df_traj_csv.iloc[-1, df_traj_csv.columns.get_loc('entropy_loss')] = -np.mean(entropy_losses)
        df_traj_csv.iloc[-1, df_traj_csv.columns.get_loc('policy_gradient_loss')] = np.mean(pg_losses)
        df_traj_csv.iloc[-1, df_traj_csv.columns.get_loc('KL_div')] = np.mean(approx_kl_divs)
        df_traj_csv.iloc[-1, df_traj_csv.columns.get_loc('explained_variance')] =  explained_var
        df_traj_csv.iloc[-1, df_traj_csv.columns.get_loc('log_std')] = th.exp(self.policy.log_std).mean().item()

        df_traj_csv.to_csv(join(process_dir, "df_traj.csv"), index=False)
        total_time_seconds.resume()

        #############################################################
    
class ModifiedFunctions_OffPolicy:
    
    def learn(
        self: SelfOffPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfOffPolicyAlgorithm:
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        ##############################MODIFICACION
        global process_dir, n_policy, total_time_seconds, callback_true, make_policy_saving
        n_policy=0
        total_time_seconds=MyTimer()
        total_time_seconds.reset()

        total_time_seconds.pause()
        # Guardar politica inicial
        if make_policy_saving:
            self.save(process_dir+'/policy'+str(n_policy)+'.zip')

        # Crear bases de datos donde ire escribiendo los datos por iteracion
        df_traj_csv=pd.DataFrame(columns=['n_policy','n_timesteps','time_seconds','traj_rewards','traj_ep_end','traj_inits',
                                          'actor_loss','critic_loss','ent_coef_loss','entropy_loss'])
        df_traj_csv.to_csv(join(process_dir, "df_traj.csv"), index=False)
        df_val_csv=pd.DataFrame(columns=['n_policy','ep_inits','ep_rewards','ep_lens','n_val_ep','elapsed_val_time'])
        df_val_csv.to_csv(join(process_dir, "df_val.csv"), index=False)
        total_time_seconds.resume()
        #################################

        callback.on_training_start(locals(), globals())

        assert self.env is not None, "You must set the environment before calling learn()"
        assert isinstance(self.train_freq, TrainFreq)  # check done in _setup_learn()

        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )

            if not rollout.continue_training:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

            ###############################MODIFICACION
            total_time_seconds.pause()
            if callback_true and make_policy_saving:
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
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        ########################MODIFICACION 
        # NOTE: esto es diferente a PPO, las listas solo se reinicializaran si los steps trasncurridos son multiplo del eval_freq puesto en callback.
        # En SAC se actualizan muchisimas mas politicas por como se define train_freq=(1,'step'), gradient_steps=1 por defecto,
        # por eso el callback no lo vamos a hacer por actualizacion (no podemos validar tantisimas politicas).
        # Entonces, los datos almacenados por interaccion hay que ir guardandolos hasta que toque validar, ahi es cuando se rellenara df_traj y se vaciaran las listas que almacenan las trajectorias.
        global total_time_seconds, process_dir
        global n_policy,df_traj
        global policy_traj_rewards,policy_traj_ep_end,policy_traj_ep_inits
        global callback_true
        total_time_seconds.pause()

        # Mirar si los steps consumidos son multiplos de la eval_freq especificada en el callback, si no son multiplos no vaciarlas
        if self.num_timesteps % (callback.eval_freq*env.num_envs) ==0:

            policy_traj_rewards=[[] for _ in range(env.num_envs)]
            policy_traj_ep_end=[[] for _ in range(env.num_envs)]
            policy_traj_ep_inits=[[] for _ in range(env.num_envs)]

        total_time_seconds.resume()
        #########################

        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            total_time_seconds.pause()#MODIFICACION: cuando no de define un callback esto no trada nada, pero cuando si se define esto tarda porque se hace validacion
            if not callback.on_step():
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)
            total_time_seconds.resume()#MODIFICACION

            ###############################MODIFICACION 
            total_time_seconds.pause()
            for i in range(env.num_envs):
                policy_traj_rewards[i].append(float(rewards[i]))
                policy_traj_ep_end[i].append(True if dones[i] else False)
                if policy_traj_ep_end[i][-1]:
                    policy_traj_ep_inits[i].append(compress_decompress_list(new_obs[i].tolist()))
            total_time_seconds.resume()
            ################################

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)  # type: ignore[arg-type]

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()

        #########################MODIFICACION
        total_time_seconds.pause()

        if self.num_timesteps % (callback.eval_freq*env.num_envs) ==0:
            callback_true=True
            with open(join(process_dir, "df_traj.csv"), 'a',newline='') as df_traj_csv:
                writer = csv.writer(df_traj_csv)
                writer.writerow([n_policy,self.num_timesteps,total_time_seconds.get_time(),compress_decompress_list(policy_traj_rewards),compress_decompress_list(policy_traj_ep_end),compress_decompress_list(policy_traj_ep_inits),
                                None,None,None,None])
        else:
            callback_true=False

        total_time_seconds.resume()
        ##########################
        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def SAC_train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        #################################MODIFICACION NOTE: a diferencia de PPO, aqui hay que calcular la entropia, porque SAC la mezcla con el coeficiente con que la optimiza
        global total_time_seconds, callback_true
        total_time_seconds.pause()
        entropy_losses=[]
        total_time_seconds.resume()
        ####################################

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            #################################MODIFICACION
            total_time_seconds.pause()
            entropy_loss=-th.mean(log_prob)
            entropy_losses.append(entropy_loss.item())
            total_time_seconds.resume()
            ####################################

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

        ####################################MODIFICACION
        # Terminar de completar datos de iteracion train guardando: 
        # - actor_loss: el equivalente a policy_loss en PPO
        # - critic_loss: SAC no calcula la divergencia de KL, este los puede servir como indice de convergencia
        # - ent_coef_loss: es la entropia pero junto con el coeficiente que se usa para ponderarla al optimizar (la guardo por si acaso)
        # - entropy_loss: la entropi a secas, esta es mas "limpia" para medir la estocasticidad de las politicas (añadida por que no se calcula por defecto en SAC)
        total_time_seconds.pause()
        if callback_true:
            df_traj_csv = pd.read_csv(join(process_dir, "df_traj.csv"))
            
            df_traj_csv.iloc[-1, df_traj_csv.columns.get_loc('actor_loss')] = np.mean(actor_losses)
            df_traj_csv.iloc[-1, df_traj_csv.columns.get_loc('critic_loss')] = np.mean(critic_losses)
            df_traj_csv.iloc[-1, df_traj_csv.columns.get_loc('ent_coef_loss')] = np.mean(ent_coef_losses)
            df_traj_csv.iloc[-1, df_traj_csv.columns.get_loc('entropy_loss')] = np.mean(entropy_losses)

            df_traj_csv.to_csv(join(process_dir, "df_traj.csv"), index=False)
        total_time_seconds.resume()

        #############################################################

class Options:
    def OffPolicy_learn_process(method,env_name,seed,total_timesteps,experiment_name,library_dir, save_policies=True, # Parametros que determinan el proceso
                      n_steps_per_env=1,n_workers=1, # Parametros que determinan la interaccion (aqui siempre n_envs_per_worker=1)
                      truth_n_workers=1, # Para el truth podemos usar un numero de workers mas elevado diferente al de por defecto para validacion
                      n_epoch=1,batch_size=256, # Parametros que determinan la actualizacion de politica
                      device='auto', vec_env_type='sequential', # Parametros que determinan el tipo de ejecucion (cpu,gpu)
                      callback=None, n_eval_ep=5, eval_freq=10000, n_eval_envs=1, deterministic_eval=False,stats_window_size=100, # tecnicas de rastreo
                      ):# Añadidas como predefinidas las variables/parametros que especifican la interaccion y la actualizacion de politica
        
        # Variables globales
        global df_traj, process_dir,make_policy_saving,all_initial_states, eval_env_name, make_eval_deterministic, make_vec_env_type, n_envs_for_truth_eval
        df_traj=[]
        make_policy_saving= save_policies
        process_dir=library_dir+'/'+experiment_name+'/process_info'
        all_initial_states=[]
        eval_env_name=env_name
        n_envs_for_truth_eval=truth_n_workers
        make_eval_deterministic=deterministic_eval
        make_vec_env_type=vec_env_type

        # Crear nuevos directorios.
        os.makedirs(process_dir)

        # Modificar funciones de librerias existentes 
        from stable_baselines3.sac import MlpPolicy, SAC
        OffPolicyAlgorithm.learn=ModifiedFunctions_OffPolicy.learn
        OffPolicyAlgorithm.collect_rollouts=ModifiedFunctions_OffPolicy.collect_rollouts
        SAC.train=ModifiedFunctions_OffPolicy.SAC_train

        if callback==True:
            EvalCallback._on_step= ModifiedFunctions_Common._on_step

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

        if method=='SAC': 
            model = SAC(MlpPolicy,
                    env, seed=seed,
                    train_freq=n_steps_per_env,batch_size=batch_size,gradient_steps=n_epoch,
                    stats_window_size=stats_window_size,
                    verbose=0,device=device)

        model.set_random_seed(seed)

        if callback==True: 
            callback=EvalCallback(eval_env,n_eval_episodes=n_eval_ep,eval_freq=eval_freq,
                                  log_path=library_dir+'/'+experiment_name,best_model_save_path=library_dir+'/'+experiment_name) 

        model.learn(total_timesteps=total_timesteps,callback=callback)


        # Guardar el modelo output
        model.save(library_dir+'/'+experiment_name+'/policy_output.zip')

    def OnPolicy_learn_process(method,env_name, # Parametros que determinan en pack
                      seed,total_timesteps,experiment_name,library_dir, save_policies=True, # Parametros que determinan el proceso
                      n_steps_per_env=2048,n_workers=1, # Parametros que determinan la interaccion (aqui siempre n_envs_per_worker=1)
                      truth_n_workers=1, # Para el truth podemos usar un numero de workers mas elevado diferente al de por defecto para validacion
                      n_epoch=10,batch_size=64, # Parametros que determinan la actualizacion de politica
                      device='auto', vec_env_type='sequential', # Parametros que determinan el tipo de ejecucion (cpu,gpu)
                      normalize=False, # Parametros relacionados con el entorno
                      policy='MlpPolicy',gae_lambda=0.95,gamma=0.99,ent_coef=0,learning_rate=0.0003,
                      clip_range=0.2,max_grad_norm=0.5, vf_coef=0.5,sde_sample_freq=-1,use_sde=False, policy_kwargs=None,# Parametros del aprendizaje
                      callback=None, n_eval_ep=5, eval_freq=10000, n_eval_envs=1, deterministic_eval=False,stats_window_size=100 # Parametros para criterios de rastreo
                      
                                            
                      ):# Añadidas como predefinidas las variables/parametros que especifican la interaccion y la actualizacion de politica.
                        # Estos parametros son los especificados en el codigo por defecto de la libreria SB3, despues para cada pack (method,env_name)
                        # ajustaremos solo aquellos que presenten una configuracion predefinida diferente segun SB3 Zoo.
        
        # Variables globales
        global df_traj, process_dir,make_policy_saving, all_initial_states, eval_env_name, make_eval_deterministic, make_vec_env_type, n_envs_for_truth_eval
        df_traj=[]
        make_policy_saving=save_policies
        process_dir=library_dir+'/'+experiment_name+'/process_info'
        all_initial_states=[]
        eval_env_name=env_name
        n_envs_for_truth_eval=truth_n_workers
        make_eval_deterministic=deterministic_eval
        make_vec_env_type=vec_env_type

        # Crear nuevos directorios.
        os.makedirs(process_dir)

        # Modificar funciones de librerias existentes TODO: cuidado cuando se usa un algortimo que no sea PPO
        from stable_baselines3.ppo import MlpPolicy, PPO
        OnPolicyAlgorithm.learn=ModifiedFunctions_OnPolicy.learn
        OnPolicyAlgorithm.collect_rollouts=ModifiedFunctions_OnPolicy.collect_rollouts
        PPO.train=ModifiedFunctions_OnPolicy.PPO_train

        if callback==True:
            EvalCallback._on_step= ModifiedFunctions_Common._on_step

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

        if method=='PPO':
            model = PPO(policy,
                        env, seed=seed,
                        n_steps=n_steps_per_env,batch_size=batch_size,n_epochs=n_epoch,
                        learning_rate=learning_rate,gamma=gamma,gae_lambda=gae_lambda,clip_range=clip_range,ent_coef=ent_coef,max_grad_norm=max_grad_norm,vf_coef=vf_coef,sde_sample_freq=sde_sample_freq,use_sde=use_sde,policy_kwargs=policy_kwargs,
                        stats_window_size=stats_window_size,
                        verbose=0,device=device)

        model.set_random_seed(seed)
        

        if callback==True: #NOTE: aqui por defecto deterministic=True, y eso hace que la politica originalmente estocastica se determinice.
                           #Aunque en SAC y PPO las politicas sean estocasticas, al validarlas nos interesa que sean deterministas.
                           #La estocasticidad es interesante durante el aprendizaje (lo favorece), no tanto en la validacion.
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
 
class PackOptions:

    def PPO_LunarLanderContinuous(seed,experiment_name,library_dir):

        ''' Configuracion tomada de: https://github.com/DLR-RM/rl-baselines3-zoo/tree/master/hyperparams
            LunarLanderContinuous-v3:
            n_envs: 16
            n_timesteps: !!float 1e6
            policy: 'MlpPolicy'
            n_steps: 1024
            batch_size: 64
            gae_lambda: 0.98
            gamma: 0.999
            n_epochs: 4
            ent_coef: 0.01

        '''
        Options.OnPolicy_learn_process(
            'PPO','LunarLanderContinuous-v3', # pack
            seed,1e6+.5*1e6,experiment_name,library_dir,save_policies=False, # learning process
            n_steps_per_env=1024,n_workers=16,truth_n_workers=16, # learning interaction
            n_epoch=4,batch_size=64, # policy update
            device='auto', vec_env_type='sequential', # execution type
            policy='MlpPolicy',gae_lambda=0.98,gamma=0.999,ent_coef=0.01, # learning process parameters
            callback=True, n_eval_ep=500, eval_freq=1024, n_eval_envs=16, deterministic_eval=True # selection criteria
            )

    def PPO_BipedalWalker(seed,experiment_name,library_dir):

        ''' Configuracion tomada de: https://github.com/DLR-RM/rl-baselines3-zoo/tree/master/hyperparams
        BipedalWalker-v3:
            normalize: true
            n_envs: 32
            n_timesteps: !!float 5e6
            policy: 'MlpPolicy'
            n_steps: 2048
            batch_size: 64
            gae_lambda: 0.95
            gamma: 0.999
            n_epochs: 10
            ent_coef: 0.0
            learning_rate: !!float 3e-4
            clip_range: 0.18

        '''
        Options.OnPolicy_learn_process(
            'PPO','BipedalWalker-v3', # pack
            seed,5e6+.5*5e6,experiment_name,library_dir,save_policies=False, # learning process
            n_steps_per_env=2048,n_workers=32, truth_n_workers=32, # learning interaction
            n_epoch=10,batch_size=64, # policy update
            device='auto', vec_env_type='sequential', # execution type
            policy='MlpPolicy',gae_lambda=0.95,gamma=0.999,ent_coef=0.0, learning_rate=3e-4, clip_range=0.18, # learning process parameters
            callback=True, n_eval_ep=500, eval_freq=2048, n_eval_envs=32, deterministic_eval=True # selection criteria
            )

    def PPO_Walker2d(seed,experiment_name,library_dir):

        ''' Configuracion tomada de: https://github.com/DLR-RM/rl-baselines3-zoo/tree/master/hyperparams
            Walker2d-v4:
                normalize: true
                n_envs: 1
                policy: 'MlpPolicy'
                n_timesteps: !!float 1e6
                batch_size: 32
                n_steps: 512
                gamma: 0.99
                learning_rate: 5.05041e-05
                ent_coef: 0.000585045
                clip_range: 0.1
                n_epochs: 20
                gae_lambda: 0.95
                max_grad_norm: 1
                vf_coef: 0.871923

        '''
        Options.OnPolicy_learn_process(
            'PPO','Walker2d-v4', # pack
            seed,1e6+.5*1e6,experiment_name,library_dir,save_policies=False, # learning process
            n_steps_per_env=512,n_workers=1,truth_n_workers=16, # learning interaction
            n_epoch=20,batch_size=32, # policy update
            device='auto', vec_env_type='sequential', # execution type
            policy='MlpPolicy',gae_lambda=0.95,gamma=0.99,ent_coef=0.000585045, learning_rate=5.05041e-05, clip_range=0.1,max_grad_norm=1, vf_coef= 0.871923,# learning process parameters
            callback=True, n_eval_ep=500, eval_freq=512, n_eval_envs=1, deterministic_eval=True # selection criteria
            )
        
    def PPO_Ant(seed,experiment_name,library_dir):
        '''
        Ant-v4: 
            normalize: true
            n_timesteps: !!float 1e6
            policy: 'MlpPolicy'
        '''

        Options.OnPolicy_learn_process(
            'PPO','Ant-v4', # pack
            seed,1e6+.5*1e6,experiment_name,library_dir,save_policies=False, # learning process
            truth_n_workers=32, # learning interaction
            device='auto', vec_env_type='sequential', # execution type
            policy='MlpPolicy', # learning process parameters
            callback=True, n_eval_ep=500, eval_freq=2048, deterministic_eval=True # selection criteria
            )

    def PPO_HalfCheetah(seed,experiment_name,library_dir):
        '''
        HalfCheetah-v4:
            normalize: true
            n_envs: 1
            policy: 'MlpPolicy'
            n_timesteps: !!float 1e6
            batch_size: 64
            n_steps: 512
            gamma: 0.98
            learning_rate: 2.0633e-05
            ent_coef: 0.000401762
            clip_range: 0.1
            n_epochs: 20
            gae_lambda: 0.92
            max_grad_norm: 0.8
            vf_coef: 0.58096
            policy_kwargs: "dict(
                                log_std_init=-2,
                                ortho_init=False,
                                activation_fn=nn.ReLU,
                                net_arch=dict(pi=[256, 256], vf=[256, 256])
                            )"
        '''

        Options.OnPolicy_learn_process(
            'PPO','HalfCheetah-v4', # pack
            seed,1e6+.5*1e6,experiment_name,library_dir,save_policies=False, # learning process
            truth_n_workers=16, batch_size=64,n_epoch=20,n_steps_per_env=512,# learning interaction
            device='auto', vec_env_type='sequential', # execution type
            policy='MlpPolicy',gamma=0.98,learning_rate=2.0633e-05,ent_coef=0.000401762,
            clip_range=0.1,gae_lambda=0.92, max_grad_norm=0.8,vf_coef=0.58096,
            policy_kwargs=dict(
                                log_std_init=-2,
                                ortho_init=False,
                                activation_fn=nn.ReLU,
                                net_arch=dict(pi=[256, 256], vf=[256, 256])
                            ), # learning process parameters
            callback=True, n_eval_ep=500, eval_freq=512, deterministic_eval=True # selection criteria
            )

    def PPO_Hopper(seed,experiment_name,library_dir):
        '''
        Hopper-v4:
            normalize: true
            n_envs: 1
            policy: 'MlpPolicy'
            n_timesteps: !!float 1e6
            batch_size: 32
            n_steps: 512
            gamma: 0.999
            learning_rate: 9.80828e-05
            ent_coef: 0.00229519
            clip_range: 0.2
            n_epochs: 5
            gae_lambda: 0.99
            max_grad_norm: 0.7
            vf_coef: 0.835671
            policy_kwargs: "dict(
                                log_std_init=-2,
                                ortho_init=False,
                                activnn.ReLU,ation_fn=
                                net_arch=dict(pi=[256, 256], vf=[256, 256])
                            )"
        '''

        Options.OnPolicy_learn_process(
            'PPO','Hopper-v4', # pack
            seed,1e6+.5*1e6,experiment_name,library_dir,save_policies=False, # learning process
            truth_n_workers=16, batch_size=32,n_epoch=5,n_steps_per_env=512,# learning interaction
            device='auto', vec_env_type='sequential', # execution type
            policy='MlpPolicy',gamma=0.999,learning_rate=9.80828e-05,ent_coef=0.00229519,
            clip_range=0.2,gae_lambda=0.99, max_grad_norm=0.7,vf_coef=0.835671,
            policy_kwargs=dict(
                                log_std_init=-2,
                                ortho_init=False,
                                activation_fn=nn.ReLU,
                                net_arch=dict(pi=[256, 256], vf=[256, 256])
                            ), # learning process parameters
            callback=True, n_eval_ep=500, eval_freq=512, deterministic_eval=True # selection criteria
            )

    def PPO_Swimmer(seed,experiment_name,library_dir):
        '''
        Swimmer-v4:
            n_timesteps: !!float 1e6
            policy: 'MlpPolicy'
            gamma: 0.9999
            n_envs: 4
            n_steps: 1024
            batch_size: 256
            learning_rate: !!float 6e-4
            gae_lambda: 0.98
        '''

        Options.OnPolicy_learn_process(
            'PPO','Swimmer-v4', # pack
            seed,1e6+.5*1e6,experiment_name,library_dir,save_policies=False, # learning process
            n_workers=4,truth_n_workers=16, batch_size=256,n_steps_per_env=1024,# learning interaction
            device='auto', vec_env_type='sequential', # execution type
            policy='MlpPolicy',gamma=0.9999,learning_rate=6e-4,gae_lambda=0.98,
            callback=True, n_eval_ep=500, eval_freq=1024, deterministic_eval=True # selection criteria
            )

#==================================================================================================
# Pruebas de funcionamiento en PC (tambien sirve como ejemplo para el cluster, se especifica
# que clases usar para cada algoritmo y diferencia de ejecucion en secuencial/paralelo)
#==================================================================================================
experiments_OnPolicy=False
experiments_OffPolicy=False
experiments_pack=False

env='Ant-v4'
seed=1
total_timesteps=2048*3
library_dir='_bender/project_SB3/outputs'

# Experimentos con OnPolicy
if experiments_OnPolicy:
    Options.OnPolicy_learn_process('PPO',env,seed,total_timesteps,'execution8',library_dir,
                        n_workers=1,
                        device='cpu',
                        callback=True,n_eval_ep=1,eval_freq=2048,n_eval_envs=2,deterministic_eval=True)

    Options.OnPolicy_learn_process('PPO',env,seed,total_timesteps,'execution2',library_dir,
                        n_workers=2,
                        device='cpu',
                        callback=True,n_eval_ep=2,eval_freq=2048,n_eval_envs=2,deterministic_eval=True)


    if __name__ == "__main__": 
        Options.OnPolicy_learn_process('PPO',env,seed,total_timesteps,'execution3',library_dir,
                            n_workers=2,vec_env_type='parallel',
                            device='cpu',
                            callback=True,n_eval_ep=2,eval_freq=2048,n_eval_envs=2,deterministic_eval=True)

    if __name__ == "__main__": 
        Options.OnPolicy_learn_process('PPO',env,seed,total_timesteps,'execution4',library_dir,
                            n_workers=1,vec_env_type='parallel',
                            device='cpu',
                            callback=True,n_eval_ep=2,eval_freq=2048,n_eval_envs=2,deterministic_eval=True)

# Experimentos con OffPolicy
if experiments_OffPolicy:
    Options.OffPolicy_learn_process('SAC',env,seed,total_timesteps,'execution5',library_dir,
                        device='cpu',
                        callback=True,n_eval_ep=2,eval_freq=1024,n_eval_envs=1,deterministic_eval=True)

    if __name__ == "__main__": 
        Options.OffPolicy_learn_process('SAC',env,seed,total_timesteps,'execution6',library_dir,save_policies=False,
                            n_workers=2,vec_env_type='parallel',
                            device='cpu',
                            callback=True,n_eval_ep=2,eval_freq=1024,n_eval_envs=2,deterministic_eval=True)

# Experimentos con packs
if experiments_pack:
    if __name__ == "__main__": 
        PackOptions.PPO_Walker2d(seed,'execution10',library_dir)




