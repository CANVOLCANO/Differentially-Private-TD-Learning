import os,sys
from icecream import ic
curPath = os.path.abspath(os.path.dirname('../../trainer/primaldual_TD_parallel.py'))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from trainer.agent import ValueAgent
import pickle 
import numpy as np
import torch
import torch.autograd as autograd
import torch.optim as optim
import tqdm
import matplotlib.pyplot as plt
import argparse
import random
import gzip
import datetime
from time import strftime

from optimizer.td import TDOptimizer
from optimizer.dptd import DPTDOptimizer

# If your OS is MacOS:
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from icecream import ic
from pathlib import Path
import wandb

def setup_seed(seed):
    """
    Set up seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(num_epoch=5, num_trajectory=1,discount=0.95, 
        optimizer_name='DPTD', 
        use_visdom=True, use_random_sample=True, 
        draw_step=10, neuron_num=50,
        compute_MSPBE=False,
        epsilon=0.1,
        value_range=[0,1],
        policy_range=[0,100],
        seed=10,
        state_dimension=4,
        record_gradient=True,
        save_model=False,
        grad_clip=10,
        traj_file_path='',
        lr_scale = 1,
        args=None
        ):

    setup_seed(seed)
    # primal
    value_agent = ValueAgent(in_features=state_dimension, mid_features=neuron_num, param_range=value_range).double()
    # dual
    omega = torch.zeros([neuron_num,1]).double()
    omega = autograd.Variable(omega, requires_grad=True)
    omega.data.uniform_(-1,1)
  
    optimizer = None
    if optimizer_name=='TD':
        optimizer = TDOptimizer(epsilon=epsilon,
                                value_range=value_range, 
                                policy_range=policy_range,
                                lr_scale = lr_scale,
                                grad_clip = grad_clip,
                                args=args)
    elif optimizer_name=='DPTD':
        optimizer = DPTDOptimizer(epsilon=epsilon,
                                value_range=value_range, 
                                policy_range=policy_range,
                                grad_clip = grad_clip,
                                args=args)
    else:
        raise Exception('Error: Optimizer type not exist.\n')

    avg_val_list = []
    avg_gradient_list = []
    T = 0
    for traj_idx in range(num_trajectory):
        with open(traj_file_path + '/%u.pkl'%traj_idx, 'rb') as f:
            trajectory = pickle.load(f)
            T += len(trajectory)
    T *= num_epoch
    
    # reset step to 0
    initial_loss = 0
    optimizer.reset_step()
    for epoch in tqdm.tqdm(range(num_epoch),desc='epochs'):
        avg_val_epoch = 0.
        avg_gradient_epoch = 0.
        for traj_idx in range(num_trajectory):
            # load trajectory
            with open(traj_file_path + '/%u.pkl'%traj_idx, 'rb') as f:
                trajectory = pickle.load(f)
            # compute variance
            optimizer.compute_noise_variance(traj_len=len(trajectory), epoch_num=num_epoch)
            avg_val_traj = 0.
            avg_gradient_traj = 0.
            for t in range(len(trajectory)):
                optimizer.zero_grad(value_agent)
                if omega.grad is not None:
                    omega.grad.data.zero_()
                if use_random_sample:
                    idx = np.random.choice(len(trajectory), 1)[0] # random sample
                else:
                    idx = t
                # get pair
                cur_state, action, next_state, reward = trajectory[idx]
                cur_state, action, next_state, reward = torch.tensor(cur_state).double(), torch.tensor(action).double(), \
                                                    torch.tensor(next_state).double(), torch.tensor(reward).double()
                # TD error
                delta = reward + discount* value_agent(next_state) - value_agent(cur_state)
                V = value_agent(cur_state)
                gradient_V_theta = autograd.grad(V, value_agent.parameters(), create_graph=True)
                if torch.__version__<'1.7.0':
                    gradient_V_theta = torch.cat([gradient_V_theta[0], gradient_V_theta[1].T], dim=1)
                else:
                    gradient_V_theta = torch.hstack([gradient_V_theta[0], gradient_V_theta[1].T]) # ?*3
                # loss
                loss_function = torch.matmul(omega.T, delta*gradient_V_theta) \
                                - 0.5*torch.square(torch.matmul(omega.T, gradient_V_theta))
                loss_function = loss_function.sum()
                if epoch == 0 and traj_idx == 0 and t == 0:
                    initial_loss = loss_function.item()
                    if args.use_wandb:
                        wandb.log({"avg_val_epoch": initial_loss})
                # gradient
                gradient_L_theta = autograd.grad(loss_function, value_agent.parameters(), create_graph=True)
                gradient_L_omega = autograd.grad(loss_function, omega, create_graph=True)
                with torch.no_grad():
                    for grad in gradient_L_theta:
                        grad[grad>grad_clip] = grad_clip
                        grad[grad<-grad_clip] = -grad_clip
                    for grad in gradient_L_omega:
                        grad[grad>grad_clip] = grad_clip
                        grad[grad<-grad_clip] = -grad_clip
                # compute step size and update
                optimizer.compute_learning_rate()
                optimizer.update_value_agent(value_agent, gradient_L_theta)
                optimizer.update_dual_variable(omega, gradient_L_omega[0])
                # save
                avg_val_traj += loss_function.item()
                if args.use_wandb and args.log_per_step_loss:
                    wandb.log({"loss_per_step": loss_function.item()})
                if record_gradient:
                    avg_gradient_traj += gradient_L_theta[0].sum()+gradient_L_theta[1].sum()
            avg_val_traj /= len(trajectory)
            avg_val_epoch += avg_val_traj
            if record_gradient:
                avg_gradient_traj /= len(trajectory)
                avg_gradient_epoch += avg_gradient_traj
        avg_val_epoch /= num_trajectory
        if record_gradient:
            avg_gradient_epoch /= num_trajectory
        avg_val_list.append(avg_val_epoch)
        if args.use_wandb:
            wandb.log({"avg_val_epoch": avg_val_epoch})
        if record_gradient:
            avg_gradient_list.append(avg_gradient_epoch.item())
    # save model
    if save_model:
        with open('./checkpoint/value_agent_%s.pkl'%optimizer_name, 'wb') as f:
            pickle.dump(value_agent, f)
    return [initial_loss] + avg_val_list, avg_gradient_list


# parse
parser = argparse.ArgumentParser(description='Differentially Private Primal Dual TD Experiments.')
parser.add_argument('--optimizer', choices=['DPTD', 'TD'], default='DPTD', type=str)
parser.add_argument('--epoch', default=100, type=int,help='Num of epoch')
parser.add_argument('--traj', default=5, type=int, help='Num of trajectories to use')
parser.add_argument('--neuron', default=50, type=int, help='Num of neurons')
parser.add_argument('--show', default=False, type=bool, help='Whether to plt show')
parser.add_argument('--save_data', default=True, type=bool, help='Whether to save data')
parser.add_argument('--epsilon', default=10, type=float, help='Param of differential privacy')
parser.add_argument('--state', default=2, type=int, help='Dimension of State Space. (CartPole=4, Acrobot=6, Atari=64)')
parser.add_argument('--test', default=False, type=bool, help='test with one seed')
parser.add_argument('--clip', default=3.0, type=float, help='value of gradient clip')
parser.add_argument('--momentum_norm', default=0.1, type=float, help='the norm used in momentum gradient estimator')
parser.add_argument('--traj_file_path', default='trajectory_acrobot', type=str, help='the file path of trajectories')
parser.add_argument('--seed_idx', default='0', type=int, help='the seed idx')
parser.add_argument('--lr_scale', default='1', type=float, help='scale the lr')
parser.add_argument('--use_wandb', default=1, type=int, help='whether to use weights&biases')
parser.add_argument('--log_per_step_loss', default=0, type=int, help='whether to log per step loss')
parser.add_argument('--noise_scale', default='1', type=float, help='scale the variance of the noises')

if __name__=='__main__':
    args = parser.parse_args()
    print(args)
    value_range = [-1, 1]
    policy_range = [-1, 1]
    seed_list = [12, 23, 34, 46, 58, 61, 79, 85, 92, 100]
    result_list = []
    gradient_list = []
    record_gradient = False
    alg_name = args.optimizer if args.optimizer != 'TD' else args.optimizer + '_{}'.format(args.lr_scale)
    prefix = '{}/{}'.format(args.traj_file_path, args.optimizer)
    fp = Path(prefix)
    if not fp.exists():
        os.makedirs(fp)
    mark = '0725_final'
    ic(mark)
    job_type = args.optimizer + ('_Eps[{}]_MN[{}]_[{}]'.format(args.epsilon, args.momentum_norm, mark) if args.optimizer == 'DPTD' else '_MN[{}]_[{}]'.format(args.momentum_norm, mark))
    ic(args.use_wandb)
    if args.use_wandb:
        wandb.init(project='2023DPTD-final',
                    name=str(args.seed_idx),
                    group=args.traj_file_path.split('/')[-1][len('trajectory_'):],
                    job_type=job_type,
                    reinit=True)
        wandb.config.update(args)
    avg_val_list, avg_gradient_list = train(num_epoch=args.epoch,
                            num_trajectory=args.traj, 
                            optimizer_name=args.optimizer, 
                            use_visdom=False, 
                            use_random_sample=True,
                            neuron_num=args.neuron,
                            compute_MSPBE=False,
                            epsilon=args.epsilon,
                            value_range=value_range,
                            policy_range=policy_range,
                            seed=seed_list[args.seed_idx],
                            state_dimension=args.state,
                            record_gradient=record_gradient,
                            grad_clip=args.clip,
                            traj_file_path = args.traj_file_path,
                            lr_scale = args.lr_scale, 
                            args=args)
    res_path = prefix + '/' + alg_name + '_seed[{}]_DP[{}]_epoch[{}]_[{}]'.format(args.seed_idx, args.epsilon, args.epoch, mark)
    ic(res_path)
    if not Path(res_path + '.npy').exists():
        max_traj_len = int(1e4)
        res = np.zeros(( max_traj_len))
        np.save(res_path, res)
    res = np.load(res_path + '.npy')
    res[:len(avg_val_list)] = np.array(avg_val_list)
    np.save(res_path, res)
