import time
from flcore.clients.clientrecon import clientRecon
from flcore.servers.serverbase import Server
from threading import Thread
import torch
import torch.nn.functional as F
from collections import OrderedDict


class Recon(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientRecon)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.layers_dict = self.clients[0]._get_layers()
        self.layers_name = list(self.layers_dict.keys())
        # saved the all cos<g_i, g_j>
        self.layer_wise_angle = OrderedDict()
        for name in self.layers_name:
            self.layer_wise_angle[name] = []


    def train(self):
        for i in range(self.global_rounds+1):
            grad_all = []
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()
                grad = client.grad2vec_list()
                grad = client.split_layer(grad_list=grad, name_dict=self.layers_dict)
                grad_all.append(grad)
                for param in client.model.parameters():
                    if param.grad is not None:
                        param.grad.zero_()
            
            # The length of the layers
            length = len(grad_all[0])

            # get the pair-wise gradients
            pair_grad = []
            for i in range(length):
                temp = []
                for j in range(self.num_join_clients):
                    temp.append(grad_all[j][i])
                temp = torch.stack(temp)
                pair_grad.append(temp)

            # get all cos<g_i, g_j>
            for i, pair in enumerate(pair_grad):
                layer_wise_cos = self.pair_cos(pair).cpu()
                self.layer_wise_angle[self.layers_name[i]].append(layer_wise_cos)

            self.receive_models()
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
            self.set_new_clients(clientRecon)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
            
    def pair_cos(self, pair):
        length = pair.size(0)

        dot_value = []
        for i in range(length - 1):
            for j in range(i + 1, length):
                dot_value.append(self.cos(pair[i], pair[j]))

        dot_value = torch.stack(dot_value).view(-1)
        return dot_value
    def cos(self, t1, t2):
        t1 = F.normalize(t1, dim=0)
        t2 = F.normalize(t2, dim=0)

        dot = (t1 * t2).sum(dim=0)

        return dot

