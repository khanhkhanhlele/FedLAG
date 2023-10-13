from flcore.clients.clientpac_rec import clientPAC_REC
from flcore.servers.serverbase import Server
from utils.data_utils import read_client_data
from threading import Thread
from collections import defaultdict
import time
import numpy as np
import random
import torch
import cvxpy as cvx
import copy
from collections import OrderedDict
import numpy as np
import torch

class PAC_REC(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientPAC_REC)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.num_classes = args.num_classes
        self.global_protos = [None for _ in range(args.num_classes)]
        
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

        self.Vars = []
        self.Hs = []
        self.uploaded_heads = []


    def train(self):
        for i in range(self.global_rounds+1):
            grad_all = []
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            self.Vars = []
            self.Hs = []
            for client in self.selected_clients:
                self.Vars.append(client.V)
                self.Hs.append(client.h)

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()

            for client in self.selected_clients:
                client.train()
                grad = client.grad2vec_list()
                grad = client.split_layer(grad_list=grad, name_dict=self.layers_dict)
                grad_all.append(grad)
                for param in client.model.parameters():
                    if param.grad is not None:
                        param.grad.zero_()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]
            
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
            
                    
                # Loops over all layers
                    # Compute number of cos < 0 -> S
                    # Sum S-conflict scores: np.sum(S)  || Sum over users' layer-wise gradient

                # Top K layers with highest score -> Get index of layers

                # -> Set of conflict layers L1     (K layers with highest scores)
                # -> Set of non-conflict layers L2 (L-K layers)
            """  """

            self.receive_protos()
            self.global_protos = proto_aggregation(self.uploaded_protos)
            self.send_protos()

            self.receive_models()
            self.aggregate_parameters()

            self.aggregate_and_send_heads()

            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break
        
        print('-'*30)
        print(self.S_score)
        top_k_items = sorted(self.S_score.items(), key=lambda x: x[1], reverse=True)[:self.top_k]
        top_k_layer = [key for key, _ in top_k_items]
        print( top_k_layer)
        print('-'*30)
        
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_model_recon(top_k_layer)

            self.Vars = []
            self.Hs = []
            for client in self.selected_clients:
                self.Vars.append(client.V)
                self.Hs.append(client.h)

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_protos()
            self.global_protos = proto_aggregation(self.uploaded_protos)
            self.send_protos()

            self.receive_models()
            self.aggregate_parameters()

            self.aggregate_and_send_heads()

            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break


        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        

    def send_protos(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_protos(self.global_protos)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_protos(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_protos = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            self.uploaded_protos.append(client.protos)

    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        # stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        # train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        # if loss == None:
        #     self.rs_train_loss.append(train_loss)
        # else:
        #     loss.append(train_loss)

        # print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        self.uploaded_heads = []
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
                self.uploaded_models.append(client.model.base)
                self.uploaded_heads.append(client.model.head)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_and_send_heads(self):
        head_weights = solve_quadratic(len(self.uploaded_ids), self.Vars, self.Hs)

        for idx, cid in enumerate(self.uploaded_ids):
            print('(Client {}) Weights of Classifier Head'.format(cid))
            print(head_weights[idx],'\n')

            if head_weights[idx] is not None:
                new_head = self.add_heads(head_weights[idx])
            else:
                new_head = self.uploaded_heads[cid]

            self.clients[cid].set_head(new_head)

    def add_heads(self, weights):
        new_head = copy.deepcopy(self.uploaded_heads[0])
        for param in new_head.parameters():
            param.data.zero_()
                    
        for w, head in zip(weights, self.uploaded_heads):
            for server_param, client_param in zip(new_head.parameters(), head.parameters()):
                server_param.data += client_param.data.clone() * w
        return new_head


# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L221
def proto_aggregation(local_protos_list):
    agg_protos_label = defaultdict(list)
    for local_protos in local_protos_list:
        for label in local_protos.keys():
            agg_protos_label[label].append(local_protos[label])

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = proto / len(proto_list)
        else:
            agg_protos_label[label] = proto_list[0].data

    return agg_protos_label


# https://github.com/JianXu95/FedPAC/blob/main/tools.py#L94
def solve_quadratic(num_users, Vars, Hs):
    device = Hs[0][0].device
    num_cls = Hs[0].shape[0] # number of classes
    d = Hs[0].shape[1] # dimension of feature representation
    avg_weight = []
    for i in range(num_users):
        # ---------------------------------------------------------------------------
        # variance ter
        v = torch.tensor(Vars, device=device)
        # ---------------------------------------------------------------------------
        # bias term
        h_ref = Hs[i]
        dist = torch.zeros((num_users, num_users), device=device)
        for j1, j2 in pairwise(tuple(range(num_users))):
            h_j1 = Hs[j1]
            h_j2 = Hs[j2]
            h = torch.zeros((d, d), device=device)
            for k in range(num_cls):
                h += torch.mm((h_ref[k]-h_j1[k]).reshape(d,1), (h_ref[k]-h_j2[k]).reshape(1,d))
            dj12 = torch.trace(h)
            dist[j1][j2] = dj12
            dist[j2][j1] = dj12

        # QP solver
        p_matrix = torch.diag(v) + dist
        p_matrix = p_matrix.cpu().numpy()  # coefficient for QP problem
        evals, evecs = torch.linalg.eig(torch.tensor(p_matrix))
        
        # for numerical stablity
        p_matrix_new = 0
        p_matrix_new = 0
        for ii in range(num_users):
            if evals[ii].real >= 0.01:
                p_matrix_new += evals[ii].real*torch.mm(evecs[:,ii].reshape(num_users,1), evecs[:,ii].reshape(1, num_users))
        p_matrix = p_matrix_new.numpy() if not np.all(np.linalg.eigvals(p_matrix)>=0.0) else p_matrix
        
        # solve QP
        alpha = 0
        eps = 1e-3
        if np.all(np.linalg.eigvals(p_matrix)>=0):
            alphav = cvx.Variable(num_users)
            obj = cvx.Minimize(cvx.quad_form(alphav, p_matrix))
            prob = cvx.Problem(obj, [cvx.sum(alphav) == 1.0, alphav >= 0])
            prob.solve()
            alpha = alphav.value
            alpha = [(i)*(i>eps) for i in alpha] # zero-out small weights (<eps)
        else:
            alpha = None # if no solution for the optimization problem, use local classifier only
        
        avg_weight.append(alpha)

    return avg_weight

# https://github.com/JianXu95/FedPAC/blob/main/tools.py#L10
def pairwise(data):
    n = len(data)
    for i in range(n):
        for j in range(i, n):
            yield (data[i], data[j])