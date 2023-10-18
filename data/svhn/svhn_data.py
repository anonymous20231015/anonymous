import numpy as np
from torchvision import datasets, transforms
from utils.util import get_root_path
from data.util import show_data
import os
import torch

def svhn_dirichlet(dataset, num_users, num_data, beta):
    '''
    :param dataset:
    :param num_users:
    :param beta:
    :return:
    '''
    np.random.seed(749)
    torch.manual_seed(749)
    min_size = 0
    min_require_size = 10
    K = 10
    num_public = 320
    l_public = 10

    labels = np.array(dataset.labels)[:num_data]
    max_value = max(labels)
    dataset.classes = [i for i in range(0, max_value + 1)]
    N = num_data

    dict_users = {}
    while min_size < min_require_size:
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

    dict_users[num_users] = np.array([])
    idxs = np.arange(len(dataset))
    # labels = np.array(dataset.targets)[:num_data]
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    num_category = np.bincount(idxs_labels[1].flatten())
    # print(num_category)
    idx_category = np.zeros(11, dtype=np.int64)
    # print(idx_category)
    for i in range(1, len(num_category) + 1):
        idx_category[i] = idx_category[i-1] + num_category[i-1]
    # print(idx_category)

    num_each = int(num_public / l_public)
    # print(num_each)
    # rand_category = np.random.choice(range(0,10), l_public, replace=False)
    rand_category = np.array(range(10))
    # print(rand_category)
    for category in rand_category:
        choices = idxs[idx_category[category] : idx_category[category+1]]
        # print(choices, len(choices))
        dict_users[num_users] = np.concatenate((dict_users[num_users], np.random.choice(choices, num_each, replace=False)), axis=0)
    # print(dict_users[num_users], len(dict_users[num_users]))
    show_data(dataset, dict_users)
    return dict_users

def get_dataset(num_data=73257, num_users=100, dirichlet=True):
    """ Returns train and val datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    data_dir = os.path.join(get_root_path(), "data", "svhn")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.SVHN(data_dir, split='train', download=True,
                                   transform=transform_train)

    test_dataset = datasets.SVHN(data_dir, split='test', download=True,
                                  transform=transform_test)

        # Sample Non-IID user data from SVHN
    if dirichlet:
        print("dirichlet distribution")
        user_groups = svhn_dirichlet(train_dataset, num_users, num_data, beta=0.5)

    return train_dataset, test_dataset, user_groups


if __name__ == '__main__':
    train_dataset, test_dataset, user_groups = get_dataset(num_data=73257, num_users=100, dirichlet=1)