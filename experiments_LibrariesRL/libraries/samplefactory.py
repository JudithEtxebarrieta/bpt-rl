import numpy as np
import torch as th
import pandas as pd
import math
from os.path import join
import sys
import os
from typing import  Tuple
import time
from collections import deque
import torch

sys.path.insert(0, os.path.abspath("libraries/sample-factory")) # Para usar misma version de sample-factory de GitHub (la que se instala con PyPI no tiene algunos ficheros)

from sample_factory.algo.learning.learner_worker import LearnerWorker
from sf_examples.mujoco.train_mujoco import register_mujoco_components, parse_mujoco_cfg, run_rl
from sample_factory.utils.utils import ensure_dir_exists, experiment_dir, log
from sample_factory.utils.typing import Config, StatusCode
from sample_factory.cfg.arguments import load_from_checkpoint
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.algo.learning.learner import Learner
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
from sample_factory.enjoy import visualize_policy_inputs, render_frame
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.huggingface.huggingface_utils import generate_model_card, generate_replay_video, push_to_hf
from sample_factory.algo.runners.runner import AlgoObserver, Runner
from sample_factory.algo.learning.learner import Learner
from libraries.commun import compress_decompress_list

class ModifiedFunctions:

    def load_checkpoint(checkpoints, device):
        if len(checkpoints) <= 0:
            log.warning("No checkpoints found")
            return None
        else:
            latest_checkpoint = checkpoints[-1]

            # extra safety mechanism to recover from spurious filesystem errors
            num_attempts = 3
            for attempt in range(num_attempts):
                # noinspection PyBroadException
                try:
                    log.warning("Loading state from checkpoint %s...", latest_checkpoint)
                    checkpoint_dict = torch.load(latest_checkpoint, map_location=device,weights_only=False) #MODIFICACION: weights_only=False para que no de error al cargar algunas politicas.
                    return checkpoint_dict
                except Exception:
                    log.exception(f"Could not load from checkpoint, attempt {attempt}")

    def on_new_training_batch(self, batch_idx: int):

        ###########################
        def my_save(self,batch_idx):
            global df_traj, total_timesteps
            last_iter=False

            # Guardar las politicas (inspirada en funciones existentes)
            checkpoint = self.learner._get_checkpoint_dict()
            process_dir=ensure_dir_exists(join(experiment_dir(cfg=self.cfg), f"process_info"))
            policy_name = f"policy_{self.training_iteration_since_resume}.pth"
            filepath = join(process_dir, policy_name)
            th.save(checkpoint, filepath)


            # Guardar las trayectorias (usa parametros identificados de interes)
            n_policy=self.training_iteration_since_resume
            n_timesteps=math.prod(self.batcher.training_batches[batch_idx]['rewards'].shape)*(n_policy+1)
            # time_seconds=None # TODO: estaria bien añadir esta columna, ya que la freq de checkpointing se mide en segundos
            traj_rewards=compress_decompress_list(self.batcher.training_batches[batch_idx]['rewards'].tolist())
            traj_ep_end=compress_decompress_list(self.batcher.training_batches[batch_idx]['dones'].tolist())
            df_traj.append([n_policy,n_timesteps,traj_rewards,traj_ep_end])
            # Comprobar si es la ultima iteracion para guardar la base de datos
            if n_timesteps>total_timesteps:
                last_iter=True
                df_traj_csv=pd.DataFrame(df_traj,columns=['n_policy','n_timesteps','traj_rewards','traj_ep_end'])
                df_traj_csv.to_csv(join(process_dir, "df_traj.csv"), index=False)
                return last_iter
            
            return last_iter

        last_iter=my_save(self,batch_idx)
        ########################


        stats = self.learner.train(self.batcher.training_batches[batch_idx])

        self.training_iteration_since_resume += 1
        self.training_batch_released.emit(batch_idx, self.training_iteration_since_resume)
        self.finished_training_iteration.emit(self.training_iteration_since_resume)
        if stats is not None:
            self.report_msg.emit(stats)

        ####################################MODIFICADO
        # Aunque se haya suerado el limite de steps en la interacion, en la ultima iteracion la politica se actualiza
        if last_iter:
            checkpoint = self.learner._get_checkpoint_dict()
            process_dir=ensure_dir_exists(join(experiment_dir(cfg=self.cfg), f"process_info"))
            policy_name = f"policy_{self.training_iteration_since_resume}.pth"
            filepath = join(process_dir, policy_name)
            th.save(checkpoint, filepath)
        #####################################

    def my_enjoy(cfg: Config) -> Tuple[StatusCode, float]:
        verbose = False

        cfg = load_from_checkpoint(cfg)

        eval_env_frameskip: int = cfg.env_frameskip if cfg.eval_env_frameskip is None else cfg.eval_env_frameskip
        assert (
            cfg.env_frameskip % eval_env_frameskip == 0
        ), f"{cfg.env_frameskip=} must be divisible by {eval_env_frameskip=}"
        render_action_repeat: int = cfg.env_frameskip // eval_env_frameskip
        cfg.env_frameskip = cfg.eval_env_frameskip = eval_env_frameskip
        log.debug(f"Using frameskip {cfg.env_frameskip} and {render_action_repeat=} for evaluation")

        cfg.num_envs = 1

        render_mode = "human"
        if cfg.save_video:
            render_mode = "rgb_array"
        elif cfg.no_render:
            render_mode = None

        env = make_env_func_batched(
            cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0), render_mode=render_mode
        )
        env_info = extract_env_info(env, cfg)

        if hasattr(env.unwrapped, "reset_on_init"):
            # reset call ruins the demo recording for VizDoom
            env.unwrapped.reset_on_init = False

        actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
        actor_critic.eval()

        device = th.device("cpu" if cfg.device == "cpu" else "cuda")
        actor_critic.model_to_device(device)

        policy_id = cfg.policy_index

        global eval_policy_from_checkpointing, eval_checkpoint_id #MODIFICADO
        if eval_policy_from_checkpointing=='True':#MODIFICADO
            if eval_checkpoint_id!=None:#MODIFICADO
                checkpoints=[Learner.checkpoint_dir(cfg, policy_id)+'/checkpoint_'+str(eval_checkpoint_id)+'.pth']
            else:#MODIFICADO
                name_prefix = dict(latest="checkpoint", best="best")[cfg.load_checkpoint_kind]
                checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(cfg, policy_id), f"{name_prefix}_*")
            

            
        else:#MODIFICADO
            checkpoints=[join(experiment_dir(cfg=cfg), "process_info")+'/policy_'+str(eval_policy_from_checkpointing)+'.pth']

        print('DIRECTORIO',checkpoints)
        checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
        actor_critic.load_state_dict(checkpoint_dict["model"])


        episode_rewards = [deque([], maxlen=100) for _ in range(env.num_agents)]
        true_objectives = [deque([], maxlen=100) for _ in range(env.num_agents)]
        num_frames = 0

        last_render_start = time.time()

        def max_frames_reached(frames):
            return cfg.max_num_frames is not None and frames > cfg.max_num_frames

        reward_list = []

        ######################################MODIFICADO
        global make_eval_deterministic
        if make_eval_deterministic:

            env.seed(0)
        ######################################

        obs, infos = env.reset()
        action_mask = obs.pop("action_mask").to(device) if "action_mask" in obs else None
        rnn_states = th.zeros([env.num_agents, get_rnn_size(cfg)], dtype=th.float32, device=device)
        episode_reward = None
        finished_episode = [False for _ in range(env.num_agents)]

        video_frames = []
        num_episodes = 0

        with th.no_grad():
            while not max_frames_reached(num_frames):
                normalized_obs = prepare_and_normalize_obs(actor_critic, obs)

                if not cfg.no_render:
                    visualize_policy_inputs(normalized_obs)
                policy_outputs = actor_critic(normalized_obs, rnn_states, action_mask=action_mask)

                # sample actions from the distribution by default
                actions = policy_outputs["actions"]

                if cfg.eval_deterministic:
                    action_distribution = actor_critic.action_distribution()
                    actions = argmax_actions(action_distribution)

                # actions shape should be [num_agents, num_actions] even if it's [1, 1]
                if actions.ndim == 1:
                    actions = unsqueeze_tensor(actions, dim=-1)
                actions = preprocess_actions(env_info, actions)

                rnn_states = policy_outputs["new_rnn_states"]

                for _ in range(render_action_repeat):
                    last_render_start = render_frame(cfg, env, video_frames, num_episodes, last_render_start)

                    obs, rew, terminated, truncated, infos = env.step(actions)
                    action_mask = obs.pop("action_mask").to(device) if "action_mask" in obs else None
                    dones = make_dones(terminated, truncated)
                    infos = [{} for _ in range(env_info.num_agents)] if infos is None else infos

                    if episode_reward is None:
                        episode_reward = rew.float().clone()
                    else:
                        episode_reward += rew.float()

                    num_frames += 1
                    if num_frames % 100 == 0:
                        log.debug(f"Num frames {num_frames}...")

                    dones = dones.cpu().numpy()
                    for agent_i, done_flag in enumerate(dones):
                        if done_flag:
                            finished_episode[agent_i] = True
                            rew = episode_reward[agent_i].item()
                            episode_rewards[agent_i].append(rew)

                            true_objective = rew
                            if isinstance(infos, (list, tuple)):
                                true_objective = infos[agent_i].get("true_objective", rew)
                            true_objectives[agent_i].append(true_objective)

                            if verbose:
                                log.info(
                                    "Episode finished for agent %d at %d frames. Reward: %.3f, true_objective: %.3f",
                                    agent_i,
                                    num_frames,
                                    episode_reward[agent_i],
                                    true_objectives[agent_i][-1],
                                )
                            rnn_states[agent_i] = th.zeros([get_rnn_size(cfg)], dtype=th.float32, device=device)
                            episode_reward[agent_i] = 0

                            if cfg.use_record_episode_statistics:
                                # we want the scores from the full episode not a single agent death (due to EpisodicLifeEnv wrapper)
                                if "episode" in infos[agent_i].keys():
                                    num_episodes += 1
                                    reward_list.append(infos[agent_i]["episode"]["r"])
                            else:
                                num_episodes += 1
                                reward_list.append(true_objective)

                    # if episode terminated synchronously for all agents, pause a bit before starting a new one
                    if all(dones):
                        render_frame(cfg, env, video_frames, num_episodes, last_render_start)
                        time.sleep(0.05)

                    if all(finished_episode):
                        finished_episode = [False] * env.num_agents
                        avg_episode_rewards_str, avg_true_objective_str = "", ""
                        for agent_i in range(env.num_agents):
                            avg_rew = np.mean(episode_rewards[agent_i])
                            avg_true_obj = np.mean(true_objectives[agent_i])

                            if not np.isnan(avg_rew):
                                if avg_episode_rewards_str:
                                    avg_episode_rewards_str += ", "
                                avg_episode_rewards_str += f"#{agent_i}: {avg_rew:.3f}"
                            if not np.isnan(avg_true_obj):
                                if avg_true_objective_str:
                                    avg_true_objective_str += ", "
                                avg_true_objective_str += f"#{agent_i}: {avg_true_obj:.3f}"

                        log.info(
                            "Avg episode rewards: %s, true rewards: %s", avg_episode_rewards_str, avg_true_objective_str
                        )
                        log.info(
                            "Avg episode reward: %.3f, avg true_objective: %.3f",
                            np.mean([np.mean(episode_rewards[i]) for i in range(env.num_agents)]),
                            np.mean([np.mean(true_objectives[i]) for i in range(env.num_agents)]),
                        )

                    # VizDoom multiplayer stuff
                    # for player in [1, 2, 3, 4, 5, 6, 7, 8]:
                    #     key = f'PLAYER{player}_FRAGCOUNT'
                    #     if key in infos[0]:
                    #         log.debug('Score for player %d: %r', player, infos[0][key])

                if num_episodes >= cfg.max_num_episodes:
                    break

        env.close()


        if cfg.save_video:
            if cfg.fps > 0:
                fps = cfg.fps
            else:
                fps = 30
            generate_replay_video(experiment_dir(cfg=cfg), video_frames, fps, cfg)

        if cfg.push_to_hub:
            generate_model_card(
                experiment_dir(cfg=cfg),
                cfg.algo,
                cfg.env,
                cfg.hf_repository,
                reward_list,
                cfg.enjoy_script,
                cfg.train_script,
            )
            push_to_hf(experiment_dir(cfg=cfg), cfg.hf_repository)


        return np.array(episode_rewards)[0],np.mean(np.array(episode_rewards)[0])


class Options:

    def learn_process(method,env,seed,total_timesteps,experiment_name,library_dir, # Parametros que determinan el proceso
                      n_steps_per_env=64, n_workers=8,n_envs_per_worker=8, # Interaccion
                      n_epoch=3,batch_size=1024, n_batches_per_epoch=8, # Actualizacion de politica
                      device='cpu', # Tipo de ejecucion
                      save_every_sec=15, keep_checkpoints=2, save_best_every_sec=5,save_best_metric='reward',save_best_after=100000, stats_avg=100 # Para seleccionar politica output
                      ):
        
        # Variables globales
        global df_traj
        df_traj=[]

        # Redefinir funciones
        LearnerWorker.on_new_training_batch=ModifiedFunctions.on_new_training_batch

        def start_learn(algo,env,seed,train_for_env_steps,experiment_name,train_dir):

            # Parametros de interes
            args = [
                # Para determinar el proceso
                f'--algo={algo}',
                f'--env={env}',
                f'--seed={seed}',
                f"--train_for_env_steps={train_for_env_steps}",

                # Directorio de almacenamiento
                f'--experiment={experiment_name}',f'--train_dir={train_dir}',

                # Para ejecutar en PC (para ejecuciones futuras en el cluster esto quitar)
                f'--device={device}',

                # Relacionados con la interaccion 
                f'--rollout={n_steps_per_env}',
                f'--num_workers={n_workers}',
                f'--num_envs_per_worker={n_envs_per_worker}',
                f'--worker_num_splits={1}',

                # Relacionados con la actualizacion de politica
                f'--batch_size={batch_size}',
                f'--num_batches_per_epoch={n_batches_per_epoch}',
                f'--num_epoch={n_epoch}',

                # Checkpointing
                f'--save_every_sec={save_every_sec}',
                f'--keep_checkpoints={keep_checkpoints}',
                f'--save_best_every_sec={save_best_every_sec}',
                f'--save_best_metric={save_best_metric}',
                f'--save_best_after={save_best_after}', 
                f'--stats_avg={stats_avg}'
                
            ]
            register_mujoco_components()
            cfg = parse_mujoco_cfg(argv=args)
            run_rl(cfg)

        if __name__ == "__main__":
            start_learn(method,env,seed,total_timesteps,experiment_name,library_dir)

    def eval_policy(env,seed,n_eval_ep,experiment_name,library_dir,
                    policy_id=False,checkpoint_id=None,load_checkpoint_kind='latest',
                    deterministic_eval=False):

        global make_eval_deterministic, eval_policy_from_checkpointing, eval_checkpoint_id
        make_eval_deterministic=deterministic_eval
        eval_policy_from_checkpointing= 'True' if str(policy_id)=='False' or checkpoint_id!=None else policy_id
        eval_checkpoint_id=checkpoint_id

        # Redefinir funciones
        Learner.load_checkpoint=ModifiedFunctions.load_checkpoint

        def start_eval(env,seed,experiment_name,train_dir):
            args = [
                    f'--env={env}',
                    f'--experiment={experiment_name}',f'--train_dir={train_dir}',
                    f'--max_num_episodes={n_eval_ep}',
                    '--no_render',
                    f'--load_checkpoint_kind={load_checkpoint_kind}'

                ]
            register_mujoco_components()
            cfg = parse_mujoco_cfg(argv=args,evaluation=True)
            status = ModifiedFunctions.my_enjoy(cfg)
            return status

        if __name__ == "__main__":
            return start_eval(env,seed,experiment_name,library_dir)


