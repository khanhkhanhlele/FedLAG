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

    def get_layers(self):
        name_list = self.model.state_dict().keys()
        layers_dict = {}
        for i, name in enumerate(name_list):
            if name not in layers_dict:
                layers_dict[name] = [i]
            else:
                layers_dict[name].append(i)

        return layers_dict
    
    def _get_layers(self):
        """
        Remove the suffix of the name of the shared layer.
        Return:
            The dictionary of shared layers: layer_dict[name]=The list of positions in the shared layers.
        """

        # parameters = self.model.parameters()
        # name_list = list(parameters.keys())
        name_list = self.model.state_dict().keys()
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

    def grad2vec_list(self):
        """
        Get parameter-wise gradients. (weight and bias are not concatenated.)
        """
        grad_list = []
        for name, param in self.model.named_parameters():
            grad = param.grad
            if grad is not None:
                grad_cur = grad.data.detach().clone().view(-1)
                grad_list.append(grad_cur)
        return grad_list

    def split_layer(self, grad_list, name_dict):
        """
        Get the layer-wise gradients. (weight and bias are concatenated.)
        """
        grad_new = []
        for key, value in name_dict.items():
            grad = [grad_list[i] for i in value]
            grad = torch.cat(grad)
            grad_new.append(grad)

        return grad_new
    
    def get_grad_dims(self):
        """
        Get the number of parameters in shared layers.
        """
        grad_dims = []
        for key, param in self.model.named_parameters():
            grad_dims.append(param.data.numel())
        return grad_dims
    
    def set_parameters_recon(self, model, layer):
        """ 
        clone parameter from layer in list layer
        """
        for model_idx, (params_model1, params_model2) in enumerate(zip(model.named_parameters(), self.model.named_parameters())):
            name_model1, param_model1 = params_model1
            name_model2, param_model2 = params_model2
            if name_model1 in layer:
                break
            param_model2 = params_model1
                

        
        # for new_param, old_param in zip(model.parameters(), self.model.parameters()):
        #     old_param.data = new_param.data.clone()
        