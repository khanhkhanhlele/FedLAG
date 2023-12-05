import copy
import numpy as np
import time
from flcore.clients.clientditto import clientDitto
from flcore.servers.serverbase import Server
from threading import Thread
import torch
from collections import OrderedDict


class Ditto_Rec(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientDitto)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        
        #recon
        self.layers_dict = self.clients[0].get_layers() #_get_layers
        self.layers_name = list(self.layers_dict.keys())
        self.grad_dims = self.clients[0].get_grad_dims()
        self.layer_wise_angle = OrderedDict()
        self.S_score = OrderedDict()
        for name in self.layers_name:
            self.S_score[name] = 0
        self.s = args.s_score
        #self.sub_method = args.sub_method
        #self.mini_rounds = args.mini_rounds
        self.mini_rounds = min(int(self.global_rounds/2),30)
        #self.mini_rounds = 30
        self.top_k = args.top_k


    def train(self):
        for i in range(self.mini_rounds+1):
            grad_all = [] #recon
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            
            for name in self.layers_name:
                self.layer_wise_angle[name] = []

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global models")
                self.evaluate()

            if i%self.eval_gap == 0:
                print("\nEvaluate personalized models")
                self.evaluate_personalized()

            for client in self.selected_clients:
                client.ptrain()
                client.train()
                grad = client.grad2vec_list()
                grad = client.split_layer(grad_list=grad, name_dict=self.layers_dict)
                grad_all.append(grad)
                for param in client.model.parameters():
                    if param.grad is not None:
                        param.grad.zero_()
                        
            # The length of the layers
            length = len(grad_all[0])       # number of layers

            """
            pair_grad = {
                Layer 1: [
                    Users 1-2, User 1-3, ... ,User (N-1)-N
                ],
                Layer 2: [
                    Users 1-2, User 1-3, ... ,User (N-1)-N
                ],
                ....
                Layer L: [
                    Users 1-2, User 1-3, ... ,User (N-1)-N
                ],
            }
            """
            # get the pair-wise gradients
            pair_grad = []
            for i in range(length):
                temp = []
                for j in range(self.num_join_clients):
                    temp.append(grad_all[j][i])
                temp = torch.stack(temp)
                pair_grad.append(temp)

            """
            layer_wise_cos = {
                Layer 1: [
                    Users 1-2, User 1-3, ... ,User (N-1)-N
                ],
                Layer 2: [
                    Users 1-2, User 1-3, ... ,User (N-1)-N
                ],
                ....
                Layer L: [
                    Users 1-2, User 1-3, ... ,User (N-1)-N
                ],
            }
            """
            # get all cos<g_i, g_j>
            for i, pair in enumerate(pair_grad):
                layer_wise_cos = self.pair_cos(pair).cpu()
                self.layer_wise_angle[self.layers_name[i]].append(layer_wise_cos)
                
            """ Calculate S-conflict scores for all users """
            
            for layer, value in self.layer_wise_angle.items():
                count = np.sum([tensor < self.s for tensor in value[0]])
                self.S_score[layer] += count
            

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break
        
        print('-'*30)
        print(self.S_score)
        top_k_items = sorted(self.S_score.items(), key=lambda x: x[1], reverse=True)[:self.top_k]
        top_k_layer = [key for key, _ in top_k_items]
        print( top_k_layer)
        print('-'*30)
        
        for i in range(self.mini_rounds+1, self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_model_recon(top_k_layer)
            #self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()
                
            if i%self.eval_gap == 0:
                print("\nEvaluate personalized models")
                self.evaluate_personalized()

            for client in self.selected_clients:
                client.ptrain()
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

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
            self.set_new_clients(clientDitto)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()


    def test_metrics_personalized(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics_personalized()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics_personalized(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics_personalized()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate_personalized(self, acc=None, loss=None):
        stats = self.test_metrics_personalized()
        stats_train = self.train_metrics_personalized()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))