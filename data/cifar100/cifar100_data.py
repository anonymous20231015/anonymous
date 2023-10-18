import numpy as np
from torchvision import datasets, transforms
from utils.util import get_root_path
from data.util import show_data
import torch
import os


def cifar_iid(dataset, num_users, num_data, num_share=0):
    """
    Sample I.I.D. client data from CIFAR100 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(num_data/num_users)
    dict_users, all_idxs = {}, [i for i in range(num_data)]
    for i in range(num_users):
        dict_users[i] = np.random.choice(all_idxs, num_items,
                                             replace=False)
        all_idxs = list(set(all_idxs) - set(dict_users[i]))

    # get shared data
    dict_users[num_users] = np.random.choice(range(num_data, len(dataset)), num_share, replace=False)

    # show data distribution
    show_data(dataset, dict_users)
    return dict_users

def cifar_dirichlet(dataset, num_users, num_data, beta):
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
    K = 100
    num_public = 3200
    l_public = 100

    labels = np.array(dataset.targets)[:num_data]
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
    idx_category = np.zeros(101, dtype=np.int64)
    # print(idx_category)
    for i in range(1, len(num_category) + 1):
        idx_category[i] = idx_category[i - 1] + num_category[i - 1]
    # print(idx_category)

    num_each = int(num_public / l_public)
    # print(num_each)
    # rand_category = np.random.choice(range(0,10), l_public, replace=False)
    rand_category = np.array(range(100))
    # print(rand_category)
    for category in rand_category:
        choices = idxs[idx_category[category]: idx_category[category + 1]]
        # print(choices, len(choices))
        dict_users[num_users] = np.concatenate(
            (dict_users[num_users], np.random.choice(choices, num_each, replace=False)), axis=0)

    show_data(dataset, dict_users)
    return dict_users

def cifar_noniid(dataset, num_users, num_data, l=20):
    """
    Sample non-I.I.D client data from CIFAR100 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards = num_users * l
    num_imgs = int(num_data / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)[:num_data]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, l, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

    # show data distribution
    show_data(dataset, dict_users)
    return dict_users

def cifar_noniid_unequal(dataset, num_users, num_data, l=2):
    """
    每个设备数量不同的 noniid
    :param dataset:
    :param num_users:
    :param num_data:
    :param num_share:
    :param l:
    :return:
    """
    idxs = np.arange(num_data)
    labels = np.array(dataset.targets)[:num_data]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # count num of each category and begin index
    num_category = np.bincount(idxs_labels[1].flatten())
    idx_category = np.zeros(len(dataset.classes), dtype=np.int64)
    for i in range(1, len(num_category)):
        idx_category[i] = idx_category[i - 1] + num_category[i - 1]

    # decide device data num for each device
    num_samples = np.random.lognormal(3, 1, num_users) + 10  # 4, 1.5 for data 1
    num_samples = num_data * num_samples / sum(num_samples)  # normalize

    dict_users = {i: np.array([]) for i in range(num_users)}
    class_per_user = np.ones(num_users) * l
    idx_train = np.zeros(len(dataset.classes), dtype=np.int64)
    for i in range(len(idx_train)):
        idx_train[i] = idx_category[i]
    for user in range(num_users):
        props = np.random.lognormal(1, 1, int(class_per_user[user]))
        props = props / sum(props)
        for j in range(int(class_per_user[user])):
            class_id = (user + j) % len(dataset.classes)
            train_sample_this_class = int(props[j] * num_samples[user]) + 1

            if idx_train[class_id] + train_sample_this_class > num_category[class_id]:
                idx_train[class_id] = idx_category[class_id]

            dict_users[user] = np.concatenate(
                (dict_users[user], idxs[idx_train[class_id]: idx_train[class_id] + train_sample_this_class]), axis=0)

            idx_train[class_id] += train_sample_this_class
    # get shared data
    # dict_users[num_users] = np.random.choice(range(num_data, len(dataset)), num_share, replace=False)



    # show data distribution
    show_data(dataset, dict_users)
    return dict_users


def get_dataset(num_data=50000, num_users=100, equal=True, unequal=False, dirichlet=False, l=20):
    """ Returns train and val datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    data_dir = os.path.join(get_root_path(), "data", "cifar100")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                   transform=transform_train)

    test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                  transform=transform_test)


    # Sample Non-IID user data from Mnist
    if unequal:
        # Chose uneuqal splits for every user
        user_groups = cifar_noniid_unequal(train_dataset, num_users, num_data, l)
        # raise NotImplementedError()
    elif equal:
        # Chose euqal splits for every user
        user_groups = cifar_noniid(train_dataset, num_users, num_data, l)
    elif dirichlet:
        user_groups = cifar_dirichlet(train_dataset, num_users, num_data, beta=0.5)

    a = 0
    for i in range(40):
        a += len(user_groups[i])
    print("total data num: ", a)
    print()
    a = 0
    for i in range(40, 80):
        a += len(user_groups[i])
    print("total data num: ", a)
    print()
    a = 0
    for i in range(80, 100):
        a += len(user_groups[i])
    print("total data num: ", a)
    print()
    return train_dataset, test_dataset, user_groups


if __name__ == '__main__':
    train_dataset, test_dataset, user_groups = get_dataset(num_data=50000, num_users=100, equal=0, unequal=0, dirichlet=1,
                                                           l=20)
