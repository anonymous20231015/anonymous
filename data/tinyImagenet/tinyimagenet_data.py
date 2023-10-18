import sys
sys.path.append('/home/work/mabeichen/FedDHAD/')
import numpy as np
from torchvision import datasets, transforms
from utils.util import get_root_path
from data.util import show_data
import os
from torchvision.datasets import DatasetFolder, ImageFolder
import torch

class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)

def tinyImageNet_dirichlet(dataset, num_users, num_data, beta):
    '''
    :param dataset:
    :param num_users:
    :param beta:
    :return:
    '''

    # np.random.seed(749)
    min_size = 0
    min_require_size = 10
    K = 200

    labels = np.array([int(s[1]) for s in dataset.samples])
    max_value = max(labels)
    dataset.classes = [i for i in range(0, max_value + 1)]
    N = num_data

    dict_users = {}
    while min_size < min_require_size:
        print(min_size)
        idx_batch = [[] for _ in range(num_users)]
        for k in range(K):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, num_users))
            proportions = np.array([p * (len(idx_j) < N / num_users) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(num_users):
        np.random.shuffle(idx_batch[j])
        dict_users[j] = idx_batch[j]

    # dict_users[num_users] = np.array([])
    # idxs = np.arange(len(dataset))
    # # labels = np.array(dataset.targets)[:num_data]
    # idxs_labels = np.vstack((idxs, labels))
    # idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    # idxs = idxs_labels[0, :]
    #
    # num_category = np.bincount(idxs_labels[1].flatten())
    # # print(num_category)
    # idx_category = np.zeros(11, dtype=np.int64)
    # # print(idx_category)
    # for i in range(1, len(num_category) + 1):
    #     idx_category[i] = idx_category[i-1] + num_category[i-1]
    # # print(idx_category)
    #
    # num_each = int(num_public / l_public)
    # # print(num_each)
    # # rand_category = np.random.choice(range(0,10), l_public, replace=False)
    # rand_category = np.array(range(10))
    # # print(rand_category)
    # for category in rand_category:
    #     choices = idxs[idx_category[category] : idx_category[category+1]]
    #     # print(choices, len(choices))
    #     dict_users[num_users] = np.concatenate((dict_users[num_users], np.random.choice(choices, num_each, replace=False)), axis=0)
    # print(dict_users[num_users], len(dict_users[num_users]))
    show_data(dataset, dict_users)
    return dict_users

def get_dataset(num_data=100000, num_users=100, dirichlet=True):
    """ Returns train and val datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    data_dir = os.path.join(get_root_path(), "data", "tinyImagenet", "tiny-imagenet-200")
    # data_dir = './tiny-imagenet-200'
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    train_dataset = ImageFolder_custom(data_dir+'/train', train=True, transform=transform)

    test_dataset = ImageFolder_custom(data_dir+'/val', train=False, transform=transform)

    # Sample Non-IID user data from Tiny-ImageNet
    if dirichlet:
        print("dirichlet distribution")
        user_groups = tinyImageNet_dirichlet(train_dataset, num_users, num_data, beta=0.5)

    return train_dataset, test_dataset, user_groups


if __name__ == '__main__':
    train_dataset, test_dataset, user_groups = get_dataset(num_data=100000, num_users=100, dirichlet=True)