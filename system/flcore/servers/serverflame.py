import time
from flcore.clients.clientflame import clientFLAME
from flcore.servers.serverbase import Server
from threading import Thread

import random
import torch
import numpy as np
import copy
import hdbscan
class FLAME(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientFLAME)
        
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []


    def train(self):
        w_glob = self.global_model.state_dict()
        w_locals = [w_glob for i in range(self.num_clients)]
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()

        
        for i in range(self.global_rounds+1):
            cos_list=[]
            local_model_vector = []
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                idx = client.id
                client.train()
                w = client.model.state_dict()
                w_locals[idx] = copy.deepcopy(w)
                
            for param in w_locals:
                # local_model_vector.append(parameters_dict_to_vector_flt_cpu(param))
                local_model_vector.append(self.parameters_dict_to_vector_flt(param))
            for i in range(len(local_model_vector)):
                cos_i = []
                for j in range(len(local_model_vector)):
                    cos_ij = 1- cos(local_model_vector[i],local_model_vector[j])
                    # cos_i.append(round(cos_ij.item(),4))
                    cos_i.append(cos_ij.item())
                cos_list.append(cos_i)
            clusterer = hdbscan.HDBSCAN(min_cluster_size=self.num_join_clients//2 + 1,min_samples=1,allow_single_cluster=True).fit(cos_list)
            print(clusterer.labels_)
            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models_flame()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

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

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientFLAME)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
            
    def receive_models_flame(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
        print(self.uploaded_weights)
    
    def parameters_dict_to_vector_flt(self, net_dict) -> torch.Tensor:
        vec = []
        for key, param in net_dict.items():
            # print(key, torch.max(param))
            if key.split('.')[-1] == 'num_batches_tracked':
                continue
            vec.append(param.view(-1))
        return torch.cat(vec)
