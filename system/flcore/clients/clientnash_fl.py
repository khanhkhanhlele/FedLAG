import torch
import torch.nn as nn
import numpy as np
import time

from flcore.clients.clientbase import Client
from utils.privacy import *


class clientNash(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.total_grads = [torch.zeros_like(param) for param in self.model.parameters()]

    def train(self):
        print(f"Client {self.id} is training.")
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for step in range(max_local_epochs):
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
                # if(step < max_local_epochs - 1 and i < len(trainloader) - 1):
                self.optimizer.step()
                    
            
            for i, param in enumerate(self.model.parameters()):
                self.total_grads[i] += param.grad.data.clone()

        self.average_grads = [param / max_local_epochs for param in self.total_grads]
        for param, grad in zip(self.model.parameters(), self.average_grads):
            param.grad.data = grad.clone()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time



    