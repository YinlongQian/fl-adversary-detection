import random
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from copy import deepcopy
device = "cuda" if torch.cuda.is_available() else "cpu"


      
def eval_op(model, loader):
    model.train()
    samples, correct = 0, 0

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            
            y_ = model(x)
            _, predicted = torch.max(y_.data, 1)

            samples += y.shape[0]
            correct += (predicted == y).sum().item()

    return correct/samples



def copy(target, source):
    for name in target:
        target[name].data = source[name].data.clone()
    
def subtract_(target, minuend, subtrahend, mode='normal'):
    for name in target:
        if mode == 'normal':
            target[name].data = minuend[name].data.clone()-subtrahend[name].data.clone()
        elif mode == 'random':
            diff = minuend[name].data.clone()-subtrahend[name].data.clone()
            diff_max = torch.max(diff)
            diff_min = torch.min(diff)
            diff_inter = diff_max - diff_min 
            target[name].data = diff_inter * torch.clamp(torch.randn_like(diff),min=0,max=1) + diff_min 
        elif mode == 'opposite': 
            target[name].data = -1 * (minuend[name].data.clone()-subtrahend[name].data.clone())

        
def flatten(source):
    return torch.cat([value.flatten() for value in source.values()])


def pairwise_angles(sources):
    angles = torch.zeros([len(sources), len(sources)])
    for i, source1 in enumerate(sources):
        for j, source2 in enumerate(sources):
            s1 = flatten(source1)
            s2 = flatten(source2)
            angles[i,j] = torch.sum(s1*s2)/(torch.norm(s1)*torch.norm(s2)+1e-12)

    return angles.numpy()


        
class FederatedTrainingDevice(object):
    def __init__(self, model_fn, data):
        self.model = model_fn().to(device)
        self.data = data
        self.W = {key : value for key, value in self.model.named_parameters()}


    def evaluate(self, loader=None):
        return eval_op(self.model, self.eval_loader if not loader else loader)
  
  
class Client(FederatedTrainingDevice):
    def __init__(self, model_fn, optimizer_fn, data, idnum, batch_size=128, train_frac=0.8, client_mode='normal'):
        super().__init__(model_fn, data)  
        self.optimizer = optimizer_fn(self.model.parameters())

        # mode - client behavior
        # 'normal': normal client
        # 'random': adversary - generate random gradients
        # 'opposite': adversary - mutiply each gradient by -1
        # 'swap': adversary - swap labels of corresponding features
        self.client_mode = client_mode
            
        self.data = data
        n_train = int(len(data)*train_frac)
        n_eval = len(data) - n_train 
        data_train, data_eval = torch.utils.data.random_split(self.data, [n_train, n_eval])

        self.train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
        self.eval_loader = DataLoader(data_eval, batch_size=batch_size, shuffle=False)
        
        self.id = idnum
        
        self.dW = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.W_old = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}
        
    def synchronize_with_server(self, server):
        copy(target=self.W, source=server.W)
    
    def compute_weight_update(self, epochs=1, loader=None):
        copy(target=self.W_old, source=self.W)
        self.optimizer.param_groups[0]["lr"]*=0.99
        train_stats = self.train_op(self.model, self.train_loader if not loader else loader, self.optimizer, epochs)
        subtract_(target=self.dW, minuend=self.W, subtrahend=self.W_old, mode=self.client_mode)
        return train_stats  

    def reset(self): 
        copy(target=self.W, source=self.W_old)

    def train_op(self, model, loader, optimizer, epochs=1):
        model.train()  
        for ep in range(epochs):
            running_loss, samples = 0.0, 0
            for x, y in loader: 
                # adversary: handle labels
                if self.client_mode == 'swap':
                    y = self.handle_labels(y, 2, 7)

                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()

                loss = torch.nn.CrossEntropyLoss()(model(x), y)
                running_loss += loss.item()*y.shape[0]
                samples += y.shape[0]

                loss.backward()

                # adversary: handle gradients
                # if self.client_mode == 'random' or self.client_mode == 'opposite':
                #     self.handle_gradients()

                optimizer.step()  

        return running_loss / samples



    
    def handle_labels(self, labels, label_1, label_2):
        labels[labels == label_1] = -1
        labels[labels == label_2] = label_1
        labels[labels == -1] = label_2

        return labels

    def handle_gradients(self):
        if self.client_mode == 'random':
            max_grad = torch.max(next(self.model.parameters()).grad)
            min_grad = torch.min(next(self.model.parameters()).grad)

            for param in self.model.parameters():
                curr_max_grad = torch.max(param.grad)
                curr_min_grad = torch.min(param.grad)

                if curr_max_grad > max_grad:
                    max_grad = curr_max_grad
                if curr_min_grad < min_grad:
                    min_grad = curr_min_grad

            diff_grad = max_grad - min_grad

            for param in self.model.parameters():
                param.grad = torch.rand(param.grad.shape, dtype=param.grad.dtype, device=param.grad.device) * diff_grad + min_grad

        elif self.client_mode == 'opposite':
            for param in self.model.parameters():
                param.grad *= -1




    
class Server(FederatedTrainingDevice):
    def __init__(self, model_fn, data, detect_mode='DBSCAN', distance_metric='euclidean'):
        super().__init__(model_fn, data)
        self.loader = DataLoader(self.data, batch_size=128, shuffle=False)
        self.model_cache = []

        self.detect_mode = detect_mode
    
    def select_clients(self, clients, frac=1.0):
        return random.sample(clients, int(len(clients)*frac)) 
    """
    def aggregate_weight_updates(self, clients):
        reduce_add_average(target=self.W, sources=[client.dW for client in clients])
    """
    def compute_pairwise_similarities(self, clients):
        return pairwise_angles([client.dW for client in clients])
  
    def cluster_clients(self, S):
        clustering = AgglomerativeClustering(affinity="precomputed", linkage="complete").fit(-S)

        c1 = np.argwhere(clustering.labels_ == 0).flatten() 
        c2 = np.argwhere(clustering.labels_ == 1).flatten() 
        return c1, c2

    def detect_adversary(self, feature_matrix, esp, min_samples, metric):
        if self.detect_mode == 'DBSCAN':

            # Noisy samples are given the label -1
            clustering = DBSCAN(eps=esp, min_samples=min_samples, metric=metric).fit(feature_matrix)
            adversary_idx = np.argwhere(clustering.labels_ == -1).flatten()
            return adversary_idx
    
    def aggregate_weight_updates(self, clients):
        self.reduce_add_average(target=self.W, sources=[client.dW for client in clients])

    def reduce_add_average(self, target, sources):
        for param_name in target:
            step = torch.mean(torch.stack([source[param_name].data for source in sources]), dim=0).clone()
            target[param_name].data += step

    def copy_weights(self, clients):
        for client in clients:
            client.W = copy(self.W)



    def aggregate_clusterwise(self, client_clusters):
        for cluster in client_clusters:
            reduce_add_average(targets=[client.W for client in cluster], 
                               sources=[client.dW for client in cluster])
            
            
    def compute_max_update_norm(self, cluster):
        return np.max([torch.norm(flatten(client.dW)).item() for client in cluster])

    
    def compute_mean_update_norm(self, cluster):
        return torch.norm(torch.mean(torch.stack([flatten(client.dW) for client in cluster]), 
                                     dim=0)).item()

    def cache_model(self, idcs, params, accuracies):
        self.model_cache += [(idcs, 
                            {name : params[name].data.clone() for name in params}, 
                            [accuracies[i] for i in idcs])]


