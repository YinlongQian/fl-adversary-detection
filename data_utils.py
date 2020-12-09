import numpy as np
from torch.utils.data import Subset
from torch.utils.data.dataset import Dataset
import torch
import torchvision.transforms as transforms

def get_default_data_transforms(name, train=True, verbose=True):
    # transforms_train = {
    # 'mnist' : transforms.Compose([
    # transforms.ToPILImage(),
    # transforms.Resize((32, 32)),
    # #transforms.RandomCrop(32, padding=4),
    # transforms.ToTensor(),
    # transforms.Normalize((0.06078,),(0.1957,))
    # ]),
    # 'fashionmnist' : transforms.Compose([
    # transforms.ToPILImage(),
    # transforms.Resize((32, 32)),
    # #transforms.RandomCrop(32, padding=4),
    # transforms.ToTensor(),
    # transforms.Normalize((0.1307,), (0.3081,))
    # ]),
    # 'cifar10' : transforms.Compose([
    # transforms.ToPILImage(),
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    # transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),#(0.24703223, 0.24348513, 0.26158784)
    # 'kws' : None
    # }
    # transforms_eval = {
    # 'mnist' : transforms.Compose([
    # transforms.ToPILImage(),
    # transforms.Resize((32, 32)),
    # transforms.ToTensor(),
    # transforms.Normalize((0.06078,),(0.1957,))
    # ]),
    # 'fashionmnist' : transforms.Compose([
    # transforms.ToPILImage(),
    # transforms.Resize((32, 32)),
    # transforms.ToTensor(),
    # transforms.Normalize((0.1307,), (0.3081,))
    # ]),
    # 'cifar10' : transforms.Compose([
    # transforms.ToPILImage(),
    # transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),#
    # 'kws' : None
    # }
    transforms_train = {
    'EMNIST': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.06078,),(0.1957,))])
    }

    transforms_eval = {
    'EMNIST': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.06078,),(0.1957,))])
    }
    if verbose:
        print("\nData preprocessing: ")
        for transformation in transforms_train[name].transforms:
            print(' -', transformation)
        print()

    return (transforms_train[name], transforms_eval[name])

class CustomImageDataset(Dataset):
    '''
    A custom Dataset class for images
    inputs : numpy array [n_data x shape]
    labels : numpy array [n_data (x 1)]
    '''
    def __init__(self, inputs, labels, transforms=None):
        assert inputs.shape[0] == labels.shape[0]
        self.inputs = torch.Tensor(inputs)
        self.labels = torch.Tensor(labels).long()
        self.transforms = transforms 

    def __getitem__(self, index):
        img, label = self.inputs[index], self.labels[index]

        if self.transforms is not None:
            img = self.transforms(img)

        return (img, label)

    def __len__(self):
        return self.inputs.shape[0]
def split_image_data(data, labels, n_clients=10, classes_per_client=10, shuffle=True, verbose=True, balancedness=None):
    '''
    Splits (data, labels) evenly among 'n_clients s.t. every client holds 'classes_per_client
    different labels
    data : [n_data x shape]
    labels : [n_data (x 1)] from 0 to n_labels
    '''
    # constants
    n_data = data.shape[0]
    n_labels = np.max(labels) + 1
  
    if balancedness >= 1.0:
        data_per_client = [n_data // n_clients]*n_clients
        data_per_client_per_class = [data_per_client[0] // classes_per_client]*n_clients
    else:
        fracs = balancedness**np.linspace(0,n_clients-1, n_clients)
        fracs /= np.sum(fracs)
        fracs = 0.1/n_clients + (1-0.1)*fracs
        data_per_client = [np.floor(frac*n_data).astype('int') for frac in fracs]

        data_per_client = data_per_client[::-1]

        data_per_client_per_class = [np.maximum(1,nd // classes_per_client) for nd in data_per_client]

    if sum(data_per_client) > n_data:
        print("Impossible Split")
        exit()
  
    # sort for labels
    data_idcs = [[] for i in range(n_labels)]
    for j, label in enumerate(labels):
        data_idcs[label] += [j]
    if shuffle:
        for idcs in data_idcs:
            np.random.shuffle(idcs)

    # split data among clients
    clients_split = []
    c = 0
    for i in range(n_clients):
        client_idcs = []
        budget = data_per_client[i]
        c = np.random.randint(n_labels)
        while budget > 0:
            take = min(data_per_client_per_class[i], len(data_idcs[c]), budget)
      
            client_idcs += data_idcs[c][:take]
            data_idcs[c] = data_idcs[c][take:]
      
            budget -= take
            c = (c + 1) % n_labels
      
        clients_split += [(data[client_idcs], labels[client_idcs])]
  
    def print_split(clients_split): 
        print("Data split:")
        for i, client in enumerate(clients_split):
            split = np.sum(client[1].reshape(1,-1)==np.arange(n_labels).reshape(-1,1), axis=1)
            print(" - Client {}: {}".format(i,split))
        print()
      
    if verbose:
        print_split(clients_split)
        
    return clients_split
def split_noniid(train_idcs, train_labels, alpha, n_clients):
    '''
    Splits a list of data indices with corresponding labels
    into subsets according to a dirichlet distribution with parameter
    alpha
    '''
    n_classes = train_labels.max()+1
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)

    class_idcs = [np.argwhere(train_labels[train_idcs]==y).flatten() 
           for y in range(n_classes)]

    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [train_idcs[np.concatenate(idcs)] for idcs in client_idcs]
  
    return client_idcs


class CustomSubset(Subset):
    '''A custom subset class with customizable data transformation'''
    def __init__(self, dataset, indices, subset_transform=None):
        super().__init__(dataset, indices)
        self.subset_transform = subset_transform
        
    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        
        if self.subset_transform:
            x = self.subset_transform(x)
      
        return x, y   