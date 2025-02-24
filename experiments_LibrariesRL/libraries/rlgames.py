import os
import numpy as np
import time
import torch 
import torch.distributed as dist
from os.path import join
import pandas as pd
from distutils.util import strtobool
import argparse, os, yaml


from libraries.commun import compress_decompress_list
from rl_games.common.a2c_common import DiscreteA2CBase, print_statistics


class ModifiedFunctions:
    def train(self):
        self.init_tensors()
        self.mean_rewards = self.last_mean_rewards = -100500
        start_time = time.perf_counter()
        total_time = 0
        rep_count = 0
        # self.frame = 0  # loading from checkpoint
        self.obs = self.env_reset()

        global df_traj, process_dir# MODIFICACION
        n_policy=0

        if self.multi_gpu:
            torch.cuda.set_device(self.local_rank)
            print("====================broadcasting parameters")
            model_params = [self.model.state_dict()]
            if self.has_central_value:
                model_params.append(self.central_value_net.state_dict())
            dist.broadcast_object_list(model_params, 0)
            self.model.load_state_dict(model_params[0])
            if self.has_central_value:
                self.central_value_net.load_state_dict(model_params[1])

        while True:
            #########################################################MODIFICACION
            # Guardar politicas
            self.save(os.path.join(process_dir, 'policy'+str(n_policy)))
            n_policy+=1
            ###########################################################

            epoch_num = self.update_epoch()
            step_time, play_time, update_time, sum_time, a_losses, c_losses, entropies, kls, last_lr, lr_mul = self.train_epoch() # COMENTARIO: aqui se hace tanto la interaccion como la actualizacion


            #########################################################MODIFICACION
            # Guardar trayectorias
                       
            num_timesteps=self.horizon_length*self.num_actors*(epoch_num+1)
            policy_traj_rewards=self.experience_buffer.tensor_dict['rewards'].tolist()
            policy_traj_ep_end=self.experience_buffer.tensor_dict['dones'].tolist()
            df_traj.append([epoch_num,num_timesteps,compress_decompress_list(policy_traj_rewards),compress_decompress_list(policy_traj_ep_end)])
            if epoch_num >= self.max_epochs:
                df_traj_csv=pd.DataFrame(df_traj,columns=['n_policy','n_timesteps','traj_rewards','traj_ep_end'])
                df_traj_csv.to_csv(join(process_dir, "df_traj.csv"), index=False)

            ###########################################################



            # cleaning memory to optimize space
            self.dataset.update_values_dict(None)
            total_time += sum_time
            curr_frames = self.curr_frames * self.world_size if self.multi_gpu else self.curr_frames
            self.frame += curr_frames
            should_exit = False


            if self.global_rank == 0:
                self.diagnostics.epoch(self, current_epoch = epoch_num)
                scaled_time = self.num_agents * sum_time
                scaled_play_time = self.num_agents * play_time

                frame = self.frame // self.num_agents

                print_statistics(self.print_stats, curr_frames, step_time, scaled_play_time, scaled_time, 
                                epoch_num, self.max_epochs, frame, self.max_frames)

                self.write_stats(total_time, epoch_num, step_time, play_time, update_time,
                                a_losses, c_losses, entropies, kls, last_lr, lr_mul, frame, 
                                scaled_time, scaled_play_time, curr_frames)

                self.algo_observer.after_print_stats(frame, epoch_num, total_time)

                if self.game_rewards.current_size > 0:
                    mean_rewards = self.game_rewards.get_mean()
                    mean_shaped_rewards = self.game_shaped_rewards.get_mean()
                    mean_lengths = self.game_lengths.get_mean()
                    self.mean_rewards = mean_rewards[0]

                    
                    for i in range(self.value_size):
                        rewards_name = 'rewards' if i == 0 else 'rewards{0}'.format(i)
                        self.writer.add_scalar(rewards_name + '/step'.format(i), mean_rewards[i], frame)
                        self.writer.add_scalar(rewards_name + '/iter'.format(i), mean_rewards[i], epoch_num)
                        self.writer.add_scalar(rewards_name + '/time'.format(i), mean_rewards[i], total_time)
                        self.writer.add_scalar('shaped_' + rewards_name + '/step'.format(i), mean_shaped_rewards[i], frame)
                        self.writer.add_scalar('shaped_' + rewards_name + '/iter'.format(i), mean_shaped_rewards[i], epoch_num)
                        self.writer.add_scalar('shaped_' + rewards_name + '/time'.format(i), mean_shaped_rewards[i], total_time)


                    self.writer.add_scalar('episode_lengths/step', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)
                    self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)

                    if self.has_self_play_config:
                        self.self_play_manager.update(self)

                    # removed equal signs (i.e. "rew=") from the checkpoint name since it messes with hydra CLI parsing
                    checkpoint_name = self.config['name'] + '_ep_' + str(epoch_num) + '_rew_' + str(mean_rewards[0])

                    if self.save_freq > 0:
                        if epoch_num % self.save_freq == 0:
                            self.save(os.path.join(self.nn_dir, 'last_' + checkpoint_name))

                    if mean_rewards[0] > self.last_mean_rewards and epoch_num >= self.save_best_after:
                        print('saving next best rewards: ', mean_rewards)
                        self.last_mean_rewards = mean_rewards[0]
                        self.save(os.path.join(self.nn_dir, self.config['name']))# TODO: se puede modificar para guardar mejor politica con otro nombre

                        if 'score_to_win' in self.config:
                            if self.last_mean_rewards > self.config['score_to_win']:
                                print('Maximum reward achieved. Network won!')
                                self.save(os.path.join(self.nn_dir, checkpoint_name))
                                should_exit = True

                if epoch_num >= self.max_epochs and self.max_epochs != -1:
                    if self.game_rewards.current_size == 0:
                        print('WARNING: Max epochs reached before any env terminated at least once')
                        mean_rewards = -np.inf

                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_ep_' + str(epoch_num) \
                        + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))
                    print('MAX EPOCHS NUM!')
                    should_exit = True

                if self.frame >= self.max_frames and self.max_frames != -1:
                    if self.game_rewards.current_size == 0:
                        print('WARNING: Max frames reached before any env terminated at least once')
                        mean_rewards = -np.inf

                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_frame_' + str(self.frame) \
                        + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))
                    print('MAX FRAMES NUM!')
                    should_exit = True

                update_time = 0

            if self.multi_gpu:
                should_exit_t = torch.tensor(should_exit, device=self.device).float()
                dist.broadcast(should_exit_t, 0)
                should_exit = should_exit_t.bool().item()

            if should_exit:
                return self.last_mean_rewards, epoch_num

class Options:

    def learn_process(method,env,seed,total_timesteps,experiment_name,experiment_param,library_dir,
                    n_steps_per_env=128, n_workers=16, # Interaccion
                    n_epoch=4,batch_size=2048, # Actualizacion de politica
                    device='cpu', # Tipo de ejecucion
                    save_best_after=4, save_frequency=4 # Para seleccionar politica output

                    ):
        
        global df_traj, process_dir
        df_traj=[]
        process_dir=library_dir+'/'+experiment_name+'/process_info'
        os.makedirs(process_dir, exist_ok=True)

        # Funciones modificadas
        DiscreteA2CBase.train=ModifiedFunctions.train
        #TODO: el env pong es Discrte, pero tengo que modificar tambien el train de Continuous (la funcion es muy parecida)

        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
     
        # Setting inicial de algunos parametros
        ap = argparse.ArgumentParser()
        ap.add_argument("--seed", type=int, default=0, required=False, 
                        help="random seed, if larger than 0 will overwrite the value in yaml config")
        ap.add_argument("-tf", "--tf", required=False, help="run tensorflow runner", action='store_true')
        ap.add_argument("-t", "--train", required=False, help="train network", action='store_true')
        ap.add_argument("-p", "--play", required=False, help="play(test) network", action='store_true')
        ap.add_argument("-c", "--checkpoint", required=False, help="path to checkpoint")
        ap.add_argument("-f", "--file", required=True, help="path to config")
        ap.add_argument("-na", "--num_actors", type=int, default=0, required=False,
                        help="number of envs running in parallel, if larger than 0 will overwrite the value in yaml config")
        ap.add_argument("-s", "--sigma", type=float, required=False, help="sets new sigma value in case if 'fixed_sigma: True' in yaml config")
        ap.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
            help="if toggled, this experiment will be tracked with Weights and Biases")
        ap.add_argument("--wandb-project-name", type=str, default="rl_games",
            help="the wandb's project name")
        ap.add_argument("--wandb-entity", type=str, default=None,
            help="the entity (team) of wandb's project")
        
        # TODO: mirar si estas dos lineas se pueden borrar, me parece que son el causante de las dos carpetas que se crean. Sin ellas da error?
        os.makedirs("nn", exist_ok=True)
        os.makedirs("runs", exist_ok=True)

        # Añadir los parametros indicados explicitamente en fichero .yaml con los parametros por defecto
        yaml_file=library_dir+'/'+experiment_param

        with open(yaml_file, "r") as file:
            data = yaml.safe_load(file)  # Cargar contenido como diccionario

        data['params']['algo']['name'] = method  # Modificar metodo por defecto
        data['params']["config"]["env_name"] = env  # Modificar env por defecto
        data['params']["seed"] = seed # Modificar seed por defecto
        data['params']["config"]['max_epochs'] = total_timesteps/(n_steps_per_env*n_workers)# Modificar total_timesteps por defecto
        data['params']["config"]['full_experiment_name'] = experiment_name # Modificar experiment_name por defecto
        data['params']["config"]['train_dir'] = library_dir # Modificar library_dir por defecto

        data['params']['config']['horizon_length'] = n_steps_per_env  # Modificar n_steps_per_env por defecto
        data['params']["config"]["num_actors"] = n_workers  # Modificar n_workers por defecto

        data['params']["config"]['mini_epochs'] = n_epoch # Modificar n_epoch por defecto
        data['params']["config"]['minibatch_size'] = batch_size # Modificar batch_size por defecto
        data['params']["config"]['seq_length'] = batch_size # TODO: entender mejor este parametro, en clase A2CBase: self.games_num = self.minibatch_size // self.seq_length


        data['params']["config"]['device'] = device # Modificar device por defecto

        data['params']["config"]['save_frequency'] = save_frequency # Modificar save_frequency por defecto
        data['params']["config"]['save_best_after'] = save_best_after # Modificar save_best_after por defecto

        with open(yaml_file, "w") as file:
            yaml.dump(data, file, default_flow_style=False)

        # Parametros para ubicacion del yalm_file e indicar que se va a entrenar
        my_args = [  
                    '--train', # Para entrenar/aprender
                    f'--file={yaml_file}'
                ]

        # Cargar todos los parametros fijados
        args = vars(ap.parse_args(args=my_args))
        config_name = args['file']

        print('Loading config: ', config_name)
        with open(config_name, 'r') as stream:
            config = yaml.safe_load(stream)

            if args['num_actors'] > 0:
                config['params']['config']['num_actors'] = args['num_actors']

            if args['seed'] > 0:
                config['params']['seed'] = args['seed']
                config['params']['config']['env_config']['seed'] = args['seed']

            from rl_games.torch_runner import Runner

            try:
                import ray
            except ImportError:
                pass
            else:
                ray.init(object_store_memory=1024*1024*1000)

            runner = Runner()
            try:
                runner.load(config)
            except yaml.YAMLError as exc:
                print(exc)

        global_rank = int(os.getenv("RANK", "0"))
        if args["track"] and global_rank == 0:
            import wandb
            wandb.init(
                project=args["wandb_project_name"],
                entity=args["wandb_entity"],
                sync_tensorboard=True,
                config=config,
                monitor_gym=True,
                save_code=True,
            )

        # Iniciar proceso
        runner.run(args)

        try:
            import ray
        except ImportError:
            pass
        else:
            ray.shutdown()

        if args["track"] and global_rank == 0:
            wandb.finish()

    def eval_policy(seed,n_eval_ep,experiment_name,experiment_param,library_dir,policy_id,
                    deterministic_eval=False):
        


        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
     
        # Setting inicial de algunos parametros
        ap = argparse.ArgumentParser()
        ap.add_argument("--seed", type=int, default=0, required=False, 
                        help="random seed, if larger than 0 will overwrite the value in yaml config")
        ap.add_argument("-tf", "--tf", required=False, help="run tensorflow runner", action='store_true')
        ap.add_argument("-t", "--train", required=False, help="train network", action='store_true')
        ap.add_argument("-p", "--play", required=False, help="play(test) network", action='store_true')
        ap.add_argument("-c", "--checkpoint", required=False, help="path to checkpoint")
        ap.add_argument("-f", "--file", required=True, help="path to config")
        ap.add_argument("-na", "--num_actors", type=int, default=0, required=False,
                        help="number of envs running in parallel, if larger than 0 will overwrite the value in yaml config")
        ap.add_argument("-s", "--sigma", type=float, required=False, help="sets new sigma value in case if 'fixed_sigma: True' in yaml config")
        ap.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
            help="if toggled, this experiment will be tracked with Weights and Biases")
        ap.add_argument("--wandb-project-name", type=str, default="rl_games",
            help="the wandb's project name")
        ap.add_argument("--wandb-entity", type=str, default=None,
            help="the entity (team) of wandb's project")
        os.makedirs("nn", exist_ok=True)
        os.makedirs("runs", exist_ok=True)

        # Añadir los parametros indicados explicitamente en fichero .yaml con los parametros por defecto
        yaml_file=library_dir+'/'+experiment_param

        with open(yaml_file, "r") as file:
            data = yaml.safe_load(file)  # Cargar contenido como diccionario

        data['params']['config']['player']['games_num'] = n_eval_ep  # Modificar n_eval_ep por defecto
        data['params']['config']['player']['render'] = False  
        data['params']['config']['device_name'] = 'cpu'  # TODO: integrarlo en los argumentos de la funcion

        with open(yaml_file, "w") as file:
            yaml.dump(data, file, default_flow_style=False)

        # Parametros para ubicacion del yalm_file e indicar que se va a validar
        checkpoint=library_dir+'/'+experiment_name+'/'+policy_id
        my_args = [  
                    '--play', # Para evaluar
                    f'--file={yaml_file}',
                    f'--checkpoint={checkpoint}'# Que politica vamos a evaluar
                ]

        # Cargar todos los parametros fijados
        args = vars(ap.parse_args(args=my_args))
        config_name = args['file']

        print('Loading config: ', config_name)
        with open(config_name, 'r') as stream:
            config = yaml.safe_load(stream)

            if args['num_actors'] > 0:
                config['params']['config']['num_actors'] = args['num_actors']

            if args['seed'] > 0:
                config['params']['seed'] = args['seed']
                config['params']['config']['env_config']['seed'] = args['seed']

            from rl_games.torch_runner import Runner

            try:
                import ray
            except ImportError:
                pass
            else:
                ray.init(object_store_memory=1024*1024*1000)

            runner = Runner()
            try:
                runner.load(config)
            except yaml.YAMLError as exc:
                print(exc)

        global_rank = int(os.getenv("RANK", "0"))
        if args["track"] and global_rank == 0:
            import wandb
            wandb.init(
                project=args["wandb_project_name"],
                entity=args["wandb_entity"],
                sync_tensorboard=True,
                config=config,
                monitor_gym=True,
                save_code=True,
            )

        # Iniciar proceso
        runner.run(args)

        try:
            import ray
        except ImportError:
            pass
        else:
            ray.shutdown()

        if args["track"] and global_rank == 0:
            wandb.finish()

        # TODO: crear un return, ahora se printea la evaluacion de los episodios en la terminal, estaria bien devolver algo



