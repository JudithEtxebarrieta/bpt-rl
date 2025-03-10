'''
Script para ejecutar job con Sample Factory en bender

NOTE: Para que lea la libreria clonada de sample-factory, modificar la siguente linea.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "libraries/sample-factory"))

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
- n_validation_ep

Valores concretos:
- device='gpu' # Para usar GPUs
- validate_policies=True # Para validar durante el proceso
- sleep_between_ep=False # Para que no se haga la pausa de 0.05 segundos entre episodios que se hace por defecto

NOTE: caracteristicas en el sbatch

- Las caracteristicas de CPU indicadas en el sbatch solo sirven para que el cluster asigne un nodo al job.
- Una vez asignado el nodo, SF no se ciñe a las caracteristicas del sbatch, coge todo lo disponible en el nodo, e.g. si el nodo tiene 96 CPU
y tu has pedido 40, SF usara los 96 CPU.
- SF para mujoco tiene una configuracion predeterminada, y define num_worker=num_envs_per_worker=8. Esto requiere 8*8=64 CPUs. 
Opcion 1. Si pedimos esto en el sbatch nos asigna el nodo3, que tiene 96 CPU y SF definira 8 workers y a cada worker le dara 96/8=12 cores.
Opcion 2. Si pedimos 40 CPUs al sbatch nos asignara el nodo2, que tiene 40 CPU y SF definira 8 workers y a cada worker le dara 40/8=5 cores.

He visto que cuando #CPU/num_workers no es entero, SF asigna todos los cores a todos los workers.

FIXME: aunque todas las politicas se guarden y cargen igual, alguna no se pueden cargar. Sale un error de torch. 
'''

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
import csv

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

import json
import bz2
import base64
import numpy as np
import os
import torch

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
        global total_time, n_val_ep, end_saving

        ###########################
        def my_save(self,batch_idx):

            total_time.pause() # Pauso el tiempo porque voy a hacer cosas adicionales a las que se hacen por defecto
            last_iter=False
            if end_saving==False:
                # Crear ficheros csv en la primera iteracion para ir escribiendo los datos
                process_dir=ensure_dir_exists(join(experiment_dir(cfg=self.cfg), f"process_info"))
                if self.training_iteration_since_resume==0:
                    df_traj_csv=pd.DataFrame(columns=['n_policy','n_timesteps','time_seconds','traj_rewards','traj_ep_end','traj_inits'])
                    df_traj_csv.to_csv(join(process_dir, "df_traj.csv"), index=False)
                    df_val_csv=pd.DataFrame(columns=['n_policy','val_time_seconds','model_load_time','ep_times','ep_between_times','ep_inits','ep_rewards','ep_lens'])
                    df_val_csv.to_csv(join(process_dir, "df_val.csv"), index=False)

                # Guardar las politicas (inspirada en funciones existentes)
                checkpoint = self.learner._get_checkpoint_dict()
                policy_name = f"policy_{self.training_iteration_since_resume}.pth"
                filepath = join(process_dir, policy_name)
                th.save(checkpoint, filepath)

                # Guardar las trayectorias (usa parametros identificados de interes)
                total_time.resume()
                time_seconds=total_time.get_time()
                total_time.pause()

                n_policy=self.training_iteration_since_resume
                n_timesteps=math.prod(self.batcher.training_batches[batch_idx]['rewards'].shape)*(n_policy+1)
                traj_rewards=self.batcher.training_batches[batch_idx]['rewards'].tolist()
                traj_ep_end=self.batcher.training_batches[batch_idx]['dones'].tolist()
                traj_states=self.batcher.training_batches[batch_idx]['obs']['obs'][:,1:].tolist()
                traj_inits=[compress_decompress_list(i.tolist()) for i in np.array(traj_states)[np.array(traj_ep_end)]]
                
                with open(join(experiment_dir(cfg=self.cfg), "process_info/df_traj.csv"), 'a',newline='') as df_traj_csv:
                    writer = csv.writer(df_traj_csv)
                    writer.writerow([n_policy,n_timesteps,time_seconds,compress_decompress_list(traj_rewards),compress_decompress_list(traj_ep_end),compress_decompress_list(traj_inits)])

                # Validar la politica y guardar datos de interes
                val_time_seconds=time.time()
                ep_rewards,ep_lens,ep_times,ep_inits,model_load_time,ep_between_times=Options.eval_policy(self.cfg.env,self.cfg.seed,n_val_ep,self.cfg.experiment,self.cfg.train_dir,policy_id=n_policy,deterministic_eval=True)
                val_time_seconds=time.time()-val_time_seconds

                with open(join(experiment_dir(cfg=self.cfg), "process_info/df_val.csv"), 'a',newline='') as df_traj_csv:
                    writer = csv.writer(df_traj_csv)
                    writer.writerow([n_policy,val_time_seconds,model_load_time,compress_decompress_list(ep_times),compress_decompress_list(ep_between_times),compress_decompress_list(ep_inits),compress_decompress_list(ep_rewards),compress_decompress_list(ep_lens)])

                # Comprobar si es la ultima iteracion
                if n_timesteps>self.cfg.train_for_env_steps:
                    last_iter=True

            return last_iter
        last_iter=my_save(self,batch_idx)
        total_time.resume()
        ########################

        stats = self.learner.train(self.batcher.training_batches[batch_idx])

        self.training_iteration_since_resume += 1
        self.training_batch_released.emit(batch_idx, self.training_iteration_since_resume)
        self.finished_training_iteration.emit(self.training_iteration_since_resume)
        if stats is not None:
            self.report_msg.emit(stats)

        ####################################MODIFICADO
        # Aunque se haya superado el limite de steps en la interacion, en la ultima iteracion la politica se actualiza
        if last_iter==True and end_saving==False:
            end_saving=True
            # Guardar la ultima politica
            checkpoint = self.learner._get_checkpoint_dict()
            process_dir=ensure_dir_exists(join(experiment_dir(cfg=self.cfg), f"process_info"))
            policy_name = f"policy_{self.training_iteration_since_resume}.pth"
            filepath = join(process_dir, policy_name)
            th.save(checkpoint, filepath)

            # Validar la politica y guardar datos de interes
            val_time_seconds=time.time()
            ep_rewards,ep_lens,ep_times,ep_inits,model_load_time,ep_between_times=Options.eval_policy(self.cfg.env,self.cfg.seed,n_val_ep,self.cfg.experiment,self.cfg.train_dir,policy_id=self.training_iteration_since_resume,deterministic_eval=True)
            val_time_seconds=time.time()-val_time_seconds

            with open(join(experiment_dir(cfg=self.cfg), "process_info/df_val.csv"), 'a',newline='') as df_traj_csv:
                writer = csv.writer(df_traj_csv)
                writer.writerow([self.training_iteration_since_resume,val_time_seconds,model_load_time,compress_decompress_list(ep_times),compress_decompress_list(ep_between_times),compress_decompress_list(ep_inits),compress_decompress_list(ep_rewards),compress_decompress_list(ep_lens)])
        #####################################

    def my_enjoy(cfg: Config) -> Tuple[StatusCode, float]:

        load_time=time.time()# MODIFICACION
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

        checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
        actor_critic.load_state_dict(checkpoint_dict["model"])
        load_time=time.time()-load_time#MODIFICACION


        episode_rewards = [deque([], maxlen=100) for _ in range(env.num_agents)]
        true_objectives = [deque([], maxlen=100) for _ in range(env.num_agents)]
        num_frames = 0
        episode_lens=[]#MODIFICACION
        episode_times=[]#MODIFICACION
        episode_between_times=[]#MODIFICACION
        episode_inits=[]#MODIFICACION

        last_render_start = time.time()

        def max_frames_reached(frames):
            return cfg.max_num_frames is not None and frames > cfg.max_num_frames

        reward_list = []

        ######################################MODIFICADO
        global make_eval_deterministic, no_sleep_between_ep
        if make_eval_deterministic:
            env.seed(0)
        ######################################
        obs, infos = env.reset()
        episode_init=compress_decompress_list(obs['obs'][0].tolist())#MODIFICACION
        action_mask = obs.pop("action_mask").to(device) if "action_mask" in obs else None
        rnn_states = th.zeros([env.num_agents, get_rnn_size(cfg)], dtype=th.float32, device=device)
        episode_reward = None
        episode_len=0#MODIFICACION
        episode_time=0#MODIFICACION

        finished_episode = [False for _ in range(env.num_agents)]

        video_frames = []
        num_episodes = 0
        with th.no_grad():
            while not max_frames_reached(num_frames):
                episode_step_time=time.time()#MODIFICACION
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

                    if episode_len==0 and len(episode_inits)>1:# MODIFICACION
                        episode_init=compress_decompress_list(obs['obs'][0].tolist())

                    last_render_start = render_frame(cfg, env, video_frames, num_episodes, last_render_start)

                    obs, rew, terminated, truncated, infos = env.step(actions)

                    action_mask = obs.pop("action_mask").to(device) if "action_mask" in obs else None
                    dones = make_dones(terminated, truncated)
                    infos = [{} for _ in range(env_info.num_agents)] if infos is None else infos

                    if episode_reward is None:
                        episode_reward = rew.float().clone()
                    else:
                        episode_reward += rew.float()
                        episode_len+=1#MODIFICACION
                        episode_time+=time.time()-episode_step_time#MODIFICACION
     
                    num_frames += 1
                    #MODIFICADO: para ahorrar tiempo no imprimiendo mensajes
                    #if num_frames % 100 == 0:
                        #log.debug(f"Num frames {num_frames}...")

                    dones = dones.cpu().numpy()
                    for agent_i, done_flag in enumerate(dones):
                        if done_flag:
                            finished_episode[agent_i] = True
                            rew = episode_reward[agent_i].item()
                            episode_rewards[agent_i].append(rew)
                            episode_lens.append(episode_len)#MODIFICACION
                            episode_times.append(episode_time)#MODIFICACION
                            episode_inits.append(episode_init)# MODIFICACION

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
                            episode_len=0#MODIFICACION
                            episode_time=0#MODIFICACION

                            if cfg.use_record_episode_statistics:
                                # we want the scores from the full episode not a single agent death (due to EpisodicLifeEnv wrapper)
                                if "episode" in infos[agent_i].keys():
                                    num_episodes += 1
                                    reward_list.append(infos[agent_i]["episode"]["r"])
                            else:
                                num_episodes += 1
                                reward_list.append(true_objective)

     
                    # if episode terminated synchronously for all agents, pause a bit before starting a new one
                    between_ep_time=time.time()#MODIFICACION
                    if all(dones):
                        render_frame(cfg, env, video_frames, num_episodes, last_render_start)
                        if no_sleep_between_ep==False:
                            time.sleep(0.05)#MODIFICACION NOTE: dejar esta pausa entre episodios realentiza muchisimo la validacion. No hacerla creo que no influye en nada


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
                        #MODIFICACION: he comentado yo estas lineas para que el imprimir no consuma tiempo
                        # log.info(
                        #     "Avg episode rewards: %s, true rewards: %s", avg_episode_rewards_str, avg_true_objective_str
                        # )
                        # log.info(
                        #     "Avg episode reward: %.3f, avg true_objective: %.3f",
                        #     np.mean([np.mean(episode_rewards[i]) for i in range(env.num_agents)]),
                        #     np.mean([np.mean(true_objectives[i]) for i in range(env.num_agents)]),
                        # )

                    # VizDoom multiplayer stuff
                    # for player in [1, 2, 3, 4, 5, 6, 7, 8]:
                    #     key = f'PLAYER{player}_FRAGCOUNT'
                    #     if key in infos[0]:
                    #         log.debug('Score for player %d: %r', player, infos[0][key])
                    episode_between_times.append(time.time()-between_ep_time)
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

        return np.array(episode_rewards)[0].tolist(), episode_lens, episode_times, episode_inits,load_time,episode_between_times
        #return np.array(episode_rewards)[0],np.mean(np.array(episode_rewards)[0])


class Options:

    def learn_process(method,env,seed,total_timesteps,experiment_name,library_dir, # Parametros que determinan el proceso
                      n_steps_per_env=64, n_workers=8,n_envs_per_worker=8, # Interaccion
                      n_epoch=3,batch_size=1024, n_batches_per_epoch=8, # Actualizacion de politica
                      device='cpu', # Tipo de ejecucion
                      save_every_sec=15, keep_checkpoints=2, save_best_every_sec=5,save_best_metric='reward',save_best_after=100000, stats_avg=100, # Para seleccionar politica output
                      validate_policies=False,n_validation_ep=100,sleep_between_ep=True, # Si se quieren validar las politicas
                      log_to_file=True,experiment_summaries_interval=3,heartbeat_interval=20,heartbeat_reporting_interval=180 # Valores por defecto que yo modifico cuando hago validacion para que el proceso no se pare o no consuma tiempo al logear
                      ):
        
        # Variables globales
        global  total_time, n_val_ep, eval_policy_from_learn_process,end_saving, no_sleep_between_ep

        total_time=MyTimer()
        total_time.reset()
        n_val_ep=n_validation_ep
        eval_policy_from_learn_process=validate_policies
        end_saving=False
        no_sleep_between_ep= not sleep_between_ep

        # Para que cuando se valida no se pare por no recibir hearbeats (si no al validar tarda mas que por defecto y piensa que se ha atascado)
        # y no tarde mas por imprimir mensajes de info.
        if validate_policies:
            heartbeat_interval=10**100
            heartbeat_reporting_interval=10**100
            experiment_summaries_interval=10**100
            log_to_file=False

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
                f'--stats_avg={stats_avg}',

                # Para evitar mensajes de informacion
                f'--log_to_file={log_to_file}',
                f'--experiment_summaries_interval={experiment_summaries_interval}',

                # Para evitar pausa por integrar la validacion en el proceso
                f'--heartbeat_interval={heartbeat_interval}',
                f'--heartbeat_reporting_interval={heartbeat_reporting_interval}'
                
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
        
        if eval_policy_from_learn_process: # Esto es porque si llamo a eval_policy desde learn_process, ya no entra en el if de main.
            return start_eval(env,seed,experiment_name,library_dir)


# Probando que el script funciona.
method='APPO'
env='mujoco_ant'
seed=1
total_timesteps=100*2*2
library_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# Ejecutar por separado los siguientes tres lear_process
Options.learn_process(method,env,seed,total_timesteps,'execution_bender1',library_dir,
                        n_steps_per_env=100,n_workers=1,n_envs_per_worker=1,
                        batch_size=100*2,n_batches_per_epoch=1,n_epoch=1,
                        validate_policies=True,n_validation_ep=2)

Options.learn_process(method,env,seed,total_timesteps,'execution_bender2',library_dir,
                        n_steps_per_env=100,n_workers=1,n_envs_per_worker=1,
                        batch_size=100*2,n_batches_per_epoch=1,n_epoch=1,
                        validate_policies=True,n_validation_ep=50)

Options.learn_process(method,env,seed,total_timesteps,'execution_bender3',library_dir,
                        n_steps_per_env=100,n_workers=1,n_envs_per_worker=1,
                        batch_size=100*2,n_batches_per_epoch=1,n_epoch=1,
                        validate_policies=True,n_validation_ep=50,sleep_between_ep=False)

# Mirar si los episodios de validacion estan pareados (he puesto determinitil_eval=True asique deberia)
df_val=pd.read_csv('_bender/project/outputs/execution_bender1/process_info/df_val.csv')
print([np.array(compress_decompress_list(i,compress=False)) for i in np.array(compress_decompress_list(df_val['ep_inits'][0],compress=False))])
print([np.array(compress_decompress_list(i,compress=False)) for i in np.array(compress_decompress_list(df_val['ep_inits'][1],compress=False))])


# Analizar donde consume tanto tiempo la validacion. Por defecto se espera 0.05 segundos entre episodio, esto realentiza la validacion.
df_val=pd.read_csv('_bender/project/outputs/execution_bender2/process_info/df_val.csv')
total_val_time=df_val['val_time_seconds'][0]
model_load_time=df_val['model_load_time'][0]
times_per_ep=np.array(compress_decompress_list(df_val['ep_times'][0],compress=False))
times_between_ep=np.array(compress_decompress_list(df_val['ep_between_times'][0],compress=False))
print('')
print('Porcentajes cuando si se hace pausa entre episodios')
print('Model load: ',model_load_time/total_val_time)
print('Episode eval: ',sum(times_per_ep)/total_val_time)
print('Episode restart: ',sum(times_between_ep)/total_val_time)

df_val=pd.read_csv('_bender/project/outputs/execution_bender3/process_info/df_val.csv')
total_val_time=df_val['val_time_seconds'][0]
model_load_time=df_val['model_load_time'][0]
times_per_ep=np.array(compress_decompress_list(df_val['ep_times'][0],compress=False))
times_between_ep=np.array(compress_decompress_list(df_val['ep_between_times'][0],compress=False))
print('')
print('Porcentajes cuando no se hace pausa entre episodios')
print('Model load: ',model_load_time/total_val_time)
print('Episode eval: ',sum(times_per_ep)/total_val_time)
print('Episode restart: ',sum(times_between_ep)/total_val_time)



