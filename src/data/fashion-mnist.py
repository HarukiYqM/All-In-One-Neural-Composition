from email import generator
from importlib import import_module
import torch.utils.data as data
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import random
import torch
import numpy as np
import collections
from PIL import Image

class FashionMNIST_truncated(data.Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.data, self.target = self.__build_truncated_dataset__()
        #print(self.data.shape)
        #print(self.target.shape)
    def __build_truncated_dataset__(self):
        cifar_dataobj = datasets.FashionMNIST(self.root, self.train, self.transform, self.target_transform, self.download)
        if self.train:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]
        return data, target

    def __getitem__(self, index):
        img, target = (self.data[index]), self.target[index]
        img = Image.fromarray(img.numpy(),mode='L')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


def gen_data_split(data, num_users, num_classes, class_partitions):
    N = data.shape[0]
    data_class_idx = {i: np.where(data== i)[0] for i in range(num_classes)}
    images_count_per_class = {i:len(data_class_idx[i]) for i in range(num_classes)}
    for data_idx in data_class_idx:
        np.random.shuffle(data_class_idx[data_idx])

    user_data_idx = collections.defaultdict(list)
    for usr_i in range(num_users):
        for c, p in zip(class_partitions['class'][usr_i], class_partitions['prob'][usr_i]):
            end_idx = int(images_count_per_class[c] * p)
            user_data_idx[usr_i].extend(data_class_idx[c][:end_idx])
            data_class_idx[c] = data_class_idx[c][end_idx:]
    for usr in user_data_idx:
        np.random.shuffle(user_data_idx[usr])
    return user_data_idx

def load_FashionMNIST_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = FashionMNIST_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = FashionMNIST_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)



def partition_data(args):    
    y_train = load_FashionMNIST_data(args.dir_data)[1]
    num_classes = 10
    classes_per_user = 10 if args.split=='iid' else 3
    num_users = args.n_agents
    
    assert (classes_per_user * num_users) % num_classes == 0, "equal classes appearance is needed"
    count_per_class = (classes_per_user * num_users) // num_classes
    class_dict = {}
    for i in range(num_classes):
        probs = np.random.uniform(1, 1, size=count_per_class)
        probs_norm = (probs / probs.sum()).tolist()
        class_dict[i] = {'count': count_per_class, 'prob': probs_norm}
    # {0:{count:10, prob=[0.1,0.1,0.1.......]}}
    class_partitions = collections.defaultdict(list)
    for i in range(num_users):
        c = []
        for _ in range(classes_per_user): #10
            class_counts = [class_dict[i]['count'] for i in range(num_classes)] #[10,10,10,10...]
            max_class_counts = np.where(np.array(class_counts) == max(class_counts))[0] #0
            c.append(np.random.choice(max_class_counts))
            class_dict[c[-1]]['count'] -= 1
        class_partitions['class'].append(c)
        class_partitions['prob'].append([class_dict[i]['prob'].pop() for i in c])
        
    agent_dataid = gen_data_split(y_train, num_users, num_classes, class_partitions)

    return agent_dataid

def get_agent_loader(args, kwargs):
    loaders_train = []
    agent_dataid = partition_data(args) #dict
    norm_mean=0.1307
    norm_std=0.3081
    g = torch.Generator()
    g.manual_seed(0)
    if not args.test_only:
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)]

        if not args.no_flip:
            transform_list.insert(1, transforms.RandomHorizontalFlip())

        transform_train = transforms.Compose(transform_list)

    for i in range(args.n_agents):
        train_ds = FashionMNIST_truncated(root=args.dir_data, dataidxs=agent_dataid[i], train=True, transform=transform_train, download=True)
        train_dl = DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=True, worker_init_fn=seed_worker,generator=g, **kwargs)
        loaders_train.append(train_dl)
    
    return loaders_train

def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

def get_loader(args, kwargs):
    norm_mean=0.1307
    norm_std=0.3081

    loader_train = None
    

    g = torch.Generator()
    g.manual_seed(0)
    if not args.test_only:
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)]

        if not args.no_flip:
            transform_list.insert(1, transforms.RandomHorizontalFlip())

        transform_train = transforms.Compose(transform_list)

        loader_train = DataLoader(
            datasets.FashionMNIST(
                root=args.dir_data,
                train=True,
                download=True,
                transform=transform_train),
            batch_size=args.batch_size, shuffle=True,worker_init_fn=seed_worker,generator=g, **kwargs
        )

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)])
    
    


    loader_test = DataLoader(
        datasets.FashionMNIST(
            root=args.dir_data,
            train=False,
            download=True,
            transform=transform_test),
        batch_size=500, shuffle=False, **kwargs
    )

    return loader_train, loader_test

