# Pytorch 
import torch
#Progress Bar
import tqdm 
# Importing required network classes
from net import InferenceNetwork,MetaLearner
# Dataset Class
from loader import Data
# Importing everything from utils
from utils import *
import logging 
import wandb
from torch.utils.tensorboard import SummaryWriter


# Main class 
class HELP:
    def __init__(self , args):
        self.args = args 
        # which mode to use training or testing 
        self.mode = args.mode
        # Choosing the metrics
        self.metrics = args.metrics
        # Checkpoint loading path 
        self.load_path = args.load_path 
        # Saving path of model weights
        self.save_path = args.save_path
        #Logs 
        # Saving the summary
        self.save_summary_steps = args.save_summary_steps
        # Saving path for the state dict 
        self.save_path = args.save_path
        # Devices
        self.meta_train_devices = args.meta_train_devices
        self.meta_valid_devics = args.meta_valid_devices
        # Number of inner tasks in a episode
        self.num_inner_tasks = args.num_inner_tasks
        # learning rate
        self.meta_lr = args.meta_lr 
        #number of episodes to train meta learner 
        self.num_episodes = args.num_episodes
        #Number of training updates for each task 
        self.num_train_updates = args.num_train_updates
        # number of eval updates 
        self.num_eval_updates = args.num_eval_updates
        # Use alpha as given in paper
        self.alpha_on = args.alpha_on 
        # Inner lr for each task 
        self.inner_lr = args.inner_lr
        # Use the second order(not understoop much)
        self.second_order = args.second_order
        # Meta-learner
        self.hw_emb_dim = args.hw_embed_dim
        self.layer_size = args.layer_size
        
        # Use hardware modulation or not
        self.z_on = args.z_on
        self.determ = args.determ
        self.kl_scaling = args.kl_scaling
        self.z_scaling = args.z_scaling
        self.mc_sampling = args.mc_sampling
        # Creating the data class with parameters that would be filled lateeeeeer
        self.data = Data(
                        args.data_path, 
                        args.meta_train_devices, 
                        args.meta_valid_devices,
                        args.num_inner_tasks, 
                        args.num_meta_train_sample,
                        args.num_samples, 
                        args.num_query)

        # Creating the inference network for getting output with respect to a task 
        self.model = MetaLearner(args.hw_embed_on,
                                 args.hw_embed_dim,
                                 args.layer_size).cuda()
        self.model_params = list(self.model.parameters())
        
        if self.alpha_on:
            self.define_task_lr_params()
            self.model_params += list(self.task_lr.values())
        else: self.task_lr = None

        if self.z_on:
            self.inference_network = InferenceNetwork(args.hw_embed_on,
                                        args.hw_embed_dim,
                                        args.layer_size,
                                        args.determ).cuda()
            self.model_params += list(self.inference_network.parameters())
    
        #Numbr of inner taks in a episode
        self.num_inner_tasks = args.num_inner_tasks
        self.loss_fn = loss_fn['mse']
        
        if self.mode == 'meta-train':
            self.meta_optimizer = torch.optim.Adam(self.model_params, lr=self.meta_lr)
            self.scheduler = None
            #Set the logger
            set_logger(os.path.join(self.save_path, 'log.txt'))
            if args.use_wandb:
                    wandb.init(entity="hayeonlee", 
                                project=args.project, 
                                name=args.exp_name, 
                                group=args.group, 
                                reinit=True)
                    wandb.config.update(args)   
                    writer = None     
            else:
                    writer = SummaryWriter(log_dir=self.save_path)
            self.log = {
                        'meta_train': Log(self.save_path, 
                                                self.save_summary_steps, 
                                                self.metrics, 
                                                self.meta_train_devices, 
                                                'meta_train', 
                                                writer, args.use_wandb),
                        'meta_valid': Log(self.save_path, 
                                                self.save_summary_steps, 
                                                self.metrics, 
                                                self.meta_valid_devices, 
                                                'meta_valid', 
                                                writer, 
                                                args.use_wandb),
                            }



    
    
    def train_single_task(self , hw_embed , xs , ys, num_updates):
        """
        hw_embed : Normalized hardware embedding 
        xs : Training set of task 
        ys : Latency labels of training set 
        """
        # Shifiting model to traning mode
        self.model.train()
        # Sending to GPU
        xs , ys = xs.cuda() , ys.cuda()
        hw_embed = hw_embed.cuda()
        # Getting the modulated initial parameter
        if self.z_on:
            params , kl, z= self.get_params_z(xs , ys , hw_embed)
        # if no modulation using the previous model parameters
        else:
            params = self.model.cloned_params()
            kl = 0.0

        # Adapted params for specific task
        adapted_params = params
        for n in range(num_updates):
            # Getting the model output on the task 
            ys_hat = self.model(xs , hw_embed , adapted_params)
            # Calculating the loss
            loss = self.loss_fn(ys_hat, ys)
            # calculating the gradients
            grads = torch.autograd.grad(
                loss, adapted_params.values(), create_graph=(self.second_order))

            #Updating the parameters of the meta Network 
            for (key, val), grad in zip(adapted_params.items(), grads):
                if self.task_lr is not None: # Meta-SGD
                    task_lr = self.task_lr[key]
                else:
                    task_lr = self.inner_lr # MAML
                adapted_params[key] = val - task_lr * grad
        return adapted_params, kl




    def meta_train(self):
        print("=> Starting training...")

        # Whether to use hardware modulator or not 
        if self.z_on:
            # If using modulator which changes theta_0 for adapting better to the task 
            self.inference_network.train()

        # Creating progress baron total number of episodes to train for
        with tqdm(total=self.num_episodes) as t:
            # Going through all the episodes
            for i_epi in range(self.num_episodes):
                # List to store adapted parameter for a task to update loss for a episode later
                adapted_state_dicts = []
                query_list = []
                # Generatin a episode which consists of a numbr of tasks
                episode = self.data.generate_episode()
                # Going through each task in the episode
                for i_task in range(self.num_inner_tasks):
                    # Getting the task speific values
                    (hw_embed , xs , ys , xq , yq , _) = episode[i_task]
                    # traning on single task 
                adapted_state_dict, kl_loss = self.train_single_task(hw_embed, xs, ys, self.num_train_updates)
                # Store adapted parameters
                # Store dataloaders for meta-update and evaluation
                adapted_state_dicts.append(adapted_state_dict)
                query_list.append((hw_embed, xq, yq))
                # Update the parameters of meta-learner
                # Compute losses with adapted parameters along with corresponding tasks
                # Updated the parameters of meta-learner using sum of the losses
                meta_loss = 0
                for i_task in range(self.num_inner_tasks):
                    # Getting the task details
                    hw_embed, xq, yq = query_list[i_task]
                    # Sending everything on cuda
                    xq, yq = xq.cuda(), yq.cuda()
                    hw_embed = hw_embed.cuda()
                    adapted_state_dict = adapted_state_dicts[i_task]
                    # Calcualting y on query set 
                    yq_hat = self.model(xq, hw_embed, adapted_state_dict)
                    loss_t = self.loss_fn(yq_hat, yq)
                    # Adding the loss for all the taks 
                    meta_loss += loss_t / float(self.num_inner_tasks) + self.kl_scaling * kl_loss
                
                # Calculating and Updating the gradients 
                self.meta_optimizer.zero_grad()
                meta_loss.backward()
                self.meta_optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step(meta_loss)


                if (i_epi + 1) % self.save_summary_steps == 0:
                    logging.info(f"Episode {i_epi+1}/{self.num_episodes}")
                    postfix = {}
                    for split in ['meta_train', 'meta_valid']:
                        msg = f"[{split.upper()}] "
                        self._test_predictor(split, i_epi)
                        self.log[split].update_epi(i_epi)
                        for m in self.metrics + ['mse_loss', 'kl_loss']:
                            v = self.log[split].avg(i_epi, m)
                            postfix[f'{split}/{m}'] = f'{v:05.3f}'
                            msg += f"{m}: {v:05.3f}; " 

                            if m == 'spearman' and  max_valid_corr < v:
                                max_valid_corr = v 
                                save_dict = {'epi': i_epi,
                                             'model': self.model.cpu().state_dict()}
                                if self.args.z_on:
                                    save_dict['inference_network'] = self.inference_network.cpu().state_dict()
                                    self.inference_network.cuda()
                                if self.args.alpha_on:
                                    save_dict['task_lr'] = {k: v.cpu() for k, v in self.task_lr.items()}
                                    for k, v in self.task_lr.items():
                                       self.task_lr[k].cuda()
                                save_path = os.path.join(self.save_path, 'checkpoint', f'help_max_corr.pt')
                                torch.save(save_dict, save_path)
                                print(f'==> save {save_path}')
                                self.model.cuda()
                        logging.info(msg)
                    t.set_postfix(postfix)
                    print('\n')
                t.update()
        self.log['meta_train'].save()
        self.log['meta_valid'].save()
        print('==> Training done')

    def test_predictor(self):
            loaded = torch.load(self.load_path)
            print(f'==> load {self.load_path}')
            if 'epi' in loaded.keys():
                epi = loaded['epi']
                print(f'==> load {epi} model..')
            self.model.load_state_dict(loaded['model'])
            if self.z_on:
                self.inference_network.load_state_dict(loaded['inference_network'])
            if self.alpha_on:
                for (k, v), (lk, lv) in zip(self.task_lr.items(), loaded['task_lr'].items()):
                    self.task_lr[k] = lv.cuda()
                
            self._test_predictor('meta_test', None)


    def _test_predictor(self, split, i_epi=None):
        save_file_path = os.path.join(self.save_path, f'test_log.txt')
        f = open(save_file_path, 'a+')

        if self.z_on:
            self.inference_network.eval()
        avg_metrics = {m: 0.0 for m in self.metrics}
        avg_metrics['mse_loss'] = 0.0

        tasks = self.data.generate_test_tasks(split) 
        for (hw_embed, xs, ys, xq, yq, device) in tasks:
            yq_hat_mean = None
            for _ in range(self.mc_sampling):
                adapted_state_dict, kl_loss = \
                    self.train_single_task(hw_embed, xs, ys, self.num_eval_updates)
                xq, yq = xq.cuda(), yq.cuda()
                hw_embed = hw_embed.cuda()
                yq_hat = self.model(xq, hw_embed, adapted_state_dict)
                if yq_hat_mean is None:
                    yq_hat_mean = yq_hat
                else:
                    yq_hat_mean += yq_hat
            yq_hat_mean = yq_hat_mean / self.args.mc_sampling
            loss = self.loss_fn(yq_hat_mean, yq)  
            
            if i_epi is not None:
                for metric in self.metrics:
                    self.log[split].update(i_epi, metric, device, 
                                            val=metrics_fn[metric](yq_hat, yq)[0])
                self.log[split].update(i_epi, 'mse_loss', device, val=loss.item())
                self.log[split].update(i_epi, 'kl_loss', device, val=kl_loss if isinstance(kl_loss, float) else kl_loss.item())
            else:
                msg = f'[{split}/{device}] '
                for m in self.metrics:
                    msg += f'{m} {metrics_fn[m](yq_hat, yq)[0]:.3f} '
                    avg_metrics[m] += metrics_fn[m](yq_hat, yq)[0]
                msg += f'MSE {loss.item():.3f}'
                avg_metrics['mse_loss'] += loss.item()
                f.write(msg+'\n')
                print(msg)

        if i_epi is None:
            nd = len(tasks)
            msg = f'[{split}/average] '
            for m in self.metrics:
                msg += f'{m} {avg_metrics[m]/nd:.3f} '
            mse_loss = avg_metrics['mse_loss']
            msg += f'MSE {mse_loss/nd:.3f} ({nd} devices)'
            f.write(msg+'\n')
            print(msg)
        f.close()
