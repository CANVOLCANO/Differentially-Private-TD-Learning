import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
import math
from icecream import ic

class TDOptimizer:
    """
    Implementation of TD Algorithm
    epsilon, delta are the privacy params
    """
    def __init__(self, 
                alpha=3,
                beta=3,
                a=1/4,
                b=3,
                chi=2, 
                eta=2,
                value_range=[-1,1],
                policy_range=[-1,1],
                epsilon=0.1, 
                delta=1e-5,
                beta_prime=0.5,
                G=1,
                grad_clip=None,
                lr_scale=None,
                args=None
                ):
        
        self.step = 0 
        self.a = a
        self.b = b
        self.beta = beta
        self.alpha = alpha
        self.chi = chi
        self.eta = eta
        self.policy_gradient = None
        self.value_gradient = []
        self.n_traj = args.traj
        # step size
        self.nu_t = None
        self.lr_scale = args.lr_scale
        # feasible set
        self.policy_range = policy_range
        self.value_range = value_range
        self.G = G
        # privacy param
        self.epsilon = 1000
        self.delta = delta
        self.beta_prime = beta_prime
        # noise variance
        self.sigma = None
        self.grad_clip = grad_clip
        self.noise_scale = args.noise_scale
        self.momentum_norm = args.momentum_norm


    @torch.no_grad() 
    def update_nu_t(self, t):
        factor = 1
        self.nu_t = factor * self.a/math.sqrt((t + self.b))


    @torch.no_grad()
    def compute_noise_variance(self, traj_len, epoch_num):
        alpha_prime = np.log(1./self.delta)/((1-self.beta_prime)*self.epsilon) + 1
        T = epoch_num*self.n_traj*traj_len
        sigma2 = 14 * np.square(self.G) * T * alpha_prime / (np.square(self.n_traj*traj_len) * self.beta_prime*self.epsilon)
        self.sigma = np.sqrt(sigma2)


    @torch.no_grad()
    def update_value_agent(self, model:nn.Module, gradients):
        if self.value_gradient==[]:
            for grad in gradients:
                # draw noise
                if self.grad_clip is not None:
                    grad[grad < -self.grad_clip] = self.grad_clip
                    grad[grad > self.grad_clip] = self.grad_clip
                noise = np.random.normal(0, self.sigma, grad.shape[0]*grad.shape[1])
                noise = torch.tensor(noise, requires_grad=False).reshape(grad.shape[0], grad.shape[1])
                grad += noise
                self.value_gradient.append(grad)
        else:
            # update gradient
            for i in range(len(self.value_gradient)):
                gradient_new = gradients[i]
                if self.grad_clip is not None:
                    gradient_new[gradient_new>self.grad_clip] = self.grad_clip
                    gradient_new[gradient_new<-self.grad_clip] = self.grad_clip
                noise = np.random.normal(0, self.sigma, gradient_new.shape[0]*gradient_new.shape[1])
                noise = torch.tensor(noise, requires_grad=False).reshape(gradient_new.shape[0], gradient_new.shape[1])
                self.value_gradient[i] *= (1-self.alpha*self.nu_t)
                self.value_gradient[i] += self.alpha*self.nu_t*gradient_new
                self.value_gradient[i] = self.norm_clip(self.value_gradient[i])
                self.value_gradient[i] += noise
        # update param
        for param, grad in zip(model.parameters(),self.value_gradient):
            if param.shape!=grad.shape:
                raise Exception("Error: Shape dismatch:", param.shape,'with', grad.shape)
            low, high = self.value_range
            # project
            param_projection = param - self.chi*grad
            param_projection[param_projection>high] = high
            param_projection[param_projection<low] = low
            # update
            param += self.nu_t*(param_projection - param)
            

    @torch.no_grad()
    def update_dual_variable(self, omega, gradients):
        if self.grad_clip is not None:
            gradients[gradients > self.grad_clip] = self.grad_clip
            gradients[gradients < -self.grad_clip] = -self.grad_clip
        # draw noise
        noise = np.random.normal(0, self.sigma, omega.shape[0])
        noise = torch.tensor(noise, requires_grad=False).unsqueeze(1)
        if self.policy_gradient==None:
            # initialize gradient
            self.policy_gradient = gradients + noise
        else:
            # update gradient
            self.policy_gradient = (1-self.beta*self.nu_t)*self.momentum_norm*self.policy_gradient + (1-(1-self.beta*self.nu_t)*self.momentum_norm)*gradients
            self.policy_gradient += noise
        # update param
        low, high = self.policy_range
        # project
        omega_projection = omega + self.eta*self.policy_gradient
        omega_projection[omega_projection>high] = high
        omega_projection[omega_projection<low] = low
        # update
        omega += self.nu_t*(omega_projection - omega)*self.lr_scale


    @torch.no_grad()
    def norm_clip(self, x):
        if self.grad_clip is not None:
            x[x > self.grad_clip] = self.grad_clip
            x[x < -self.grad_clip] = -self.grad_clip
        return x


    @torch.no_grad()
    def reset_step(self):
        """
        reset step to 0
        """
        self.step = 0


    @torch.no_grad()
    def zero_grad(self, model:nn.Module):
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.zero_()


    @torch.no_grad()
    def compute_learning_rate(self):
        self.step += 1
        self.update_nu_t(self.step)