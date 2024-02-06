import copy
import re
import torch
import numpy as np
from flcore.clients.clientnash_fl import clientNash
from flcore.servers.serverbase import Server
from threading import Thread
import torch
import time

import cvxpy as cp


class NashFL(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientNash)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

        #upload grad vs param
        self.uploaded_grads = []
        self.uploaded_params = []
        self.grads_flatten = []

        #variable for solve alpha 
        self.G = torch.eye(self.current_num_join_clients)
        
        self.prvs_alpha = np.ones(self.current_num_join_clients, dtype=np.float32)
        self.normalization_factor = np.ones((1,))
        self.init_gtg = np.eye(self.current_num_join_clients, dtype=np.float32)
        self.prvs_alpha = np.ones(self.current_num_join_clients, dtype=np.float32)
        self.optim_niter = 1000
        self.alpha = []


    def get_param_grad(self):
        self.uploaded_params = []
        self.uploaded_grads = []    
        for model in self.uploaded_models:
            temp_params = []
            temp_grads = []
            for param in model.parameters():
                temp_params.append(param.data.clone())
                temp_grads.append(param.grad.clone())
            self.uploaded_params.append(temp_params)
            self.uploaded_grads.append(temp_grads)


    def aggregate_grads(self):
        self.grads_flatten = []
        for grad in self.uploaded_grads:
            grad_flat = torch.cat([g.view(-1) for g in grad])
            self.grads_flatten.append(grad_flat)
        self.G = torch.stack(self.grads_flatten, dim=0)

    def _init_optim_problem(self):
        self.alpha_param = cp.Variable(self.current_num_join_clients, nonneg=True)
        self.prvs_alpha_param = cp.Parameter(self.current_num_join_clients, value=self.prvs_alpha)
        self.G_param = cp.Parameter(
            shape=(self.current_num_join_clients, self.num_clients), value=self.init_gtg
        )
        self.normalization_factor_param = cp.Parameter(
            shape=(1,), value=np.array([1.0])
        )

        self.phi_alpha = self._calc_phi_alpha_linearization()

        G_alpha = self.G_param @ self.alpha_param

        constraints = []

        for i in range(self.num_clients):
            constraints.append(
                -cp.log(self.alpha_param[i] * self.normalization_factor_param)
                - cp.log(G_alpha[i])
                <= 0
            )

        obj = cp.Minimize(
            cp.sum(G_alpha) + self.phi_alpha / self.normalization_factor_param
        )

        self.prob = cp.Problem(obj, constraints)
        

    def _calc_phi_alpha_linearization(self):
        G_prvs_alpha = self.G_param @ self.prvs_alpha_param
        prvs_phi_tag = 1 / self.prvs_alpha_param + (1 / G_prvs_alpha) @ self.G_param
        phi_alpha = prvs_phi_tag @ (self.alpha_param - self.prvs_alpha_param)
        return phi_alpha

    def _stop_criteria(self, gtg, alpha_t):
        # print(np.linalg.norm(gtg @ alpha_t - 1 / (alpha_t + 1e-10)))
        return (
            (self.alpha_param.value is None)
            or (np.linalg.norm(gtg @ alpha_t - 1 / (alpha_t + 1e-10)) < 1e-3)
            or (
                np.linalg.norm(self.alpha_param.value - self.prvs_alpha_param.value)
                < 1e-6
            )
        )
    
    def solve(self, gtg: np.array): 
        self.G_param.value = gtg
        self.normalization_factor_param.value = self.normalization_factor

        alpha_t = self.prvs_alpha

        for _ in range(self.optim_niter):
            self.alpha_param.value = alpha_t
            self.prvs_alpha_param.value = alpha_t

            try:
                self.prob.solve(solver=cp.ECOS, warm_start=True, max_iters=100)
            except:
                self.alpha_param.value = self.prvs_alpha_param.value

            if self._stop_criteria(gtg, alpha_t):
                break

            alpha_t = self.alpha_param.value

        if alpha_t is not None:
            self.prvs_alpha = alpha_t

        return self.prvs_alpha

    def get_alpha(self):
        self._init_optim_problem() 
        GTG = torch.matmul(self.G, self.G.t())
        # print(GTG.shape)
        self.normalization_factor = (
            torch.norm(GTG).detach().cpu().numpy().reshape((1,))
        )
        GTG = GTG / self.normalization_factor.item()

        alpha = self.solve(GTG.cpu().detach().numpy())
        # alpha = torch.from_numpy(alpha)
        alpha_sum = np.sum(alpha)
        alpha = alpha / alpha_sum
        return alpha
    
    #update global model
    
    def aggregate_parameters_nash(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.alpha, self.uploaded_models):
            self.add_parameters_nash(w, client_model)
        #check parameters after aggregate
        # print("parameters after aggregate")
        # for param in self.global_model.parameters():
        #     print(param.data)
        

    def add_parameters_nash(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w
        


    def check(self):
        GTG = torch.matmul(self.G, self.G.t())
        vt = GTG @ self.alpha
        vp = 1 / (self.alpha + 1e-10)
        # print("vt is ", vt)
        # print("vp is ",vp)
        print("alpha is :", self.alpha)

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()   # select clients
            
            #check parameters before send global model
            print("parameter of a selected client before send global model")
            for name, param in self.selected_clients[0].model.named_parameters():
                print(name, param.data[2])
                break
            
            self.send_models()                        # send global model

            #print only first parameters of global model
            print("parameters of global model before aggregate")
            for name, param in self.global_model.named_parameters():
                print(name, param.data[2])
                break
            
            #check success of send global model
            print("parameter of a selected client after send global model")
            for name, param in self.selected_clients[0].model.named_parameters():
                print(name, param.data[2])
                break
            
            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()                    # evaluate global model

            for client in self.selected_clients:
                client.train()                # train selected clients

            print("parameters after trainning of a selected client")
            for name, param in self.selected_clients[0].model.named_parameters():
                print(name, param.data[2])
                break
            
            
            self.receive_models()              # receive models from clients
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)        
            
            self.get_param_grad()            # get param vs grad each client
            self.aggregate_grads()      # aggregate grad to matrix G


            self.alpha = self.get_alpha()       # get alpha

            self.check()               # check alpha

            self.aggregate_parameters_nash()        #update global model

            print("parameters after aggregate")
            for param in self.global_model.parameters():
                print(param.data[2])
                break

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()            # save results
        self.save_global_model()    # save global model


