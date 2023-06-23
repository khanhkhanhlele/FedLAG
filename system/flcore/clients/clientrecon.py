import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.privacy import *
from collections import OrderedDict
import os.path as osp

class clientRecon(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.layers_dict = self._get_layers()
        self.layers_name = list(self.layers_dict.keys())
        print(self.layers_name)
        # saved the all cos<g_i, g_j>
        self.layer_wise_angle = OrderedDict()
        for name in self.layers_name:
            self.layer_wise_angle[name] = []
    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()

        # differential privacy
        if self.privacy:
            self.model, self.optimizer, trainloader, privacy_engine = \
                initialize_dp(self.model, self.optimizer, trainloader, self.dp_sigma)
        
        start_time = time.time()

        max_local_steps = self.local_epochs
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        if self.privacy:
            eps, DELTA = get_dp_params(privacy_engine)
            print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")

        print(a for a in self.model.parameters())
    def _get_layers(self):
        """
        Remove the suffix of the name of the shared layer.
        Return:
            The dictionary of shared layers: layer_dict[name]=The list of positions in the shared layers.
        """

        shared_parameters = self.network.shared_parameters()

        name_list = list(shared_parameters.keys())
        layers_dict = {}
        for i, name in enumerate(name_list):
            if '.weight' in name:
                name = name.replace('.weight', '')
            elif '.bias' in name:
                name = name.replace('.bias', '')

            if name not in layers_dict:
                layers_dict[name] = [i]
            else:
                layers_dict[name].append(i)

        return layers_dict

    def cagrad(self, grads, alpha=0.5, rescale=1):
        """
        Copy from the implementation of CAGrad: https://github.com/Cranial-XIX/CAGrad
        """
        GG = grads.t().mm(grads).cpu()  # [num_tasks, num_tasks]
        g0_norm = (GG.mean() + 1e-8).sqrt()  # norm of the average gradient

        n_tasks = self.n_tasks
        x_start = np.ones(n_tasks) / n_tasks
        bnds = tuple((0, 1) for x in x_start)
        cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})
        A = GG.numpy()
        b = x_start.copy()
        c = (alpha * g0_norm + 1e-8).item()

        def objfn(x):
            return (x.reshape(1, n_tasks).dot(A).dot(b.reshape(n_tasks, 1)) + c * np.sqrt(
                x.reshape(1, n_tasks).dot(A).dot(x.reshape(n_tasks, 1)) + 1e-8)).sum()

        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        ww = torch.Tensor(w_cpu).to(grads.device)
        gw = (grads * ww.view(1, -1)).sum(1)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm + 1e-8)
        g = grads.mean(1) + lmbda * gw
        if rescale == 0:
            return g
        elif rescale == 1:
            return g / (1 + alpha ** 2)
        else:
            return g / (1 + alpha)

    def grad2vec_list(self):
        """
        Get parameter-wise gradients. (weight and bias are not concatenated.)
        """
        grad_list = []
        for name, param in self.network.shared_parameters().items():
            grad = param.grad
            if grad is not None:
                grad_cur = grad.data.detach().clone().view(-1)
                grad_list.append(grad_cur)
        return grad_list

    def __split_layer(self, grad_list, name_dict):
        """
        Get the layer-wise gradients. (weight and bias are concatenated.)
        """
        grad_new = []
        for key, value in name_dict.items():
            grad = [grad_list[i] for i in value]
            grad = torch.cat(grad)
            grad_new.append(grad)

        return grad_new

    def manipulate_grad(self, losses):
        # store the gradients of each task
        grad_all = []

        for i, task in enumerate(self.tasks):
            if i < self.n_tasks:
                losses[task].backward(retain_graph=True)
            else:
                losses[task].backward()

            self._grad2vec(i)

            grad = self.grad2vec_list()
            grad = self.__split_layer(grad_list=grad, name_dict=self.layers_dict)

            grad_all.append(grad)
            self.network.zero_grad_shared_modules()

        # get the update gradient after gradient manipulation
        if self.sub_method == 'Baseline':
            g = torch.sum(self.grads, dim=1) / self.n_tasks
        elif self.sub_method == 'CAGrad':
            g = self.cagrad(self.grads, self.alpha, rescale=1)
        else:
            raise NotImplementedError

        # The length of the layers
        length = len(grad_all[0])

        # get the pair-wise gradients
        pair_grad = []
        for i in range(length):
            temp = []
            for j in range(self.n_tasks):
                temp.append(grad_all[j][i])
            temp = torch.stack(temp)
            pair_grad.append(temp)

        # get all cos<g_i, g_j>
        for i, pair in enumerate(pair_grad):
            layer_wise_cos = pair_cos(pair).cpu()
            self.layer_wise_angle[self.layers_name[i]].append(layer_wise_cos)

        self.overwrite_grad(g)

        return

    def save(self, path, name, epoch, iterations, seed):
        saved_dict = {'cos': self.layer_wise_angle}
        torch.save(saved_dict, osp.join(path, f'{seed}_{self.sub_method}_ep{epoch}_lw_cos.pt'))
        torch.save(self.network.state_dict(), osp.join(path, f'{seed}_{name}_ep{epoch}_iter{iterations}.pt'))
