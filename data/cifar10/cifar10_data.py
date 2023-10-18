import numpy as np
from torchvision import datasets, transforms
from utils.util import get_root_path
from data.util import show_data
import os
import torch

def cifar_iid(dataset, num_users, num_data, num_share=0):
    """
    Sample I.I.D. client data from CIFAR10 dataset
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
    K = 10
    num_public = 320
    l_public = 10

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

def cifar_noniid(dataset, num_users, num_data, num_share=0, l=2, l_share=10):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
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
    devices have different nums of data and categories
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
    idx_category = np.zeros(10, dtype=np.int64)
    for i in range(1, len(num_category)):
        idx_category[i] = idx_category[i - 1] + num_category[i - 1]

    # decide device data num for each device
    num_samples = np.random.lognormal(3, 1, num_users) + 10  # 4, 1.5 for data 1
    num_samples = num_data * num_samples / sum(num_samples)  # normalize

    dict_users = {i: np.array([]) for i in range(num_users)}
    class_per_user = np.ones(num_users) * l
    idx_train = np.zeros(10, dtype=np.int64)
    for i in range(len(idx_train)):
        idx_train[i] = idx_category[i]
    for user in range(num_users):
        props = np.random.lognormal(1, 1, int(class_per_user[user]))
        props = props / sum(props)
        for j in range(int(class_per_user[user])):
            class_id = (user + j) % 10
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

def cifar_noniid_unbalance(dataset, num_users, num_data):
    """
    devices have different nums of data and categories, {40% l=1} {40% l=2} {20% l=10}
    :param dataset:
    :param num_users:
    :param num_data:
    :param num_share:
    :param l:
    :return:
    """
    user_list = [int(num_users*0.4), int(num_users*0.4), int(num_users*0.2)]
    data_list = [int(num_data*0.4), int(num_data*0.4), int(num_data*0.2)]
    l_list = [1, 2, 10]

    dict_users = {}
    n=0
    for num_users, num_data, l in zip(user_list, data_list, l_list):
        print("l={}: user num: {} data num: {}".format(l,num_users,num_data))
        idxs = np.arange(num_data)
        labels = np.array(dataset.targets)[:num_data]

        # sort labels
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]

        # count num of each category and begin index
        num_category = np.bincount(idxs_labels[1].flatten())
        idx_category = np.zeros(10, dtype=np.int64)
        for i in range(1, len(num_category)):
            idx_category[i] = idx_category[i - 1] + num_category[i - 1]

        # decide device data num for each device
        num_samples = np.random.lognormal(3, 1, num_users) + 10  # 4, 1.5 for data 1
        num_samples = num_data * num_samples / sum(num_samples)  # normalize

        if n==0:
            t_dict_users = {i: np.array([]) for i in range(num_users)}
        elif n==1:
            t_dict_users = {i+40: np.array([]) for i in range(num_users)}
        else:
            t_dict_users = {i+80: np.array([]) for i in range(num_users)}
        class_per_user = np.ones(num_users) * l
        idx_train = np.zeros(10, dtype=np.int64)
        for i in range(len(idx_train)):
            idx_train[i] = idx_category[i]
        for user in range(num_users):
            props = np.random.lognormal(1, 1, int(class_per_user[user]))
            props = props / sum(props)
            for j in range(int(class_per_user[user])):
                class_id = (user + j) % 10
                train_sample_this_class = int(props[j] * num_samples[user]) + 1

                if idx_train[class_id] + train_sample_this_class > num_category[class_id]:
                    idx_train[class_id] = idx_category[class_id]

                t_dict_users[user+n*40] = np.concatenate(
                    (t_dict_users[user+n*40], idxs[idx_train[class_id]: idx_train[class_id] + train_sample_this_class]), axis=0)

                idx_train[class_id] += train_sample_this_class
        n+=1
        dict_users.update(t_dict_users)

    # show data distribution
    show_data(dataset, dict_users)
    return dict_users

def cifar_noniid_e(dataset, num_users, num_data):
    """
    devices have different nums of data and categories, (ratio * num_users) number of users have all categories.
    :param dataset:
    :param num_users:
    :param num_data:
    :param l:
    :param ratio:
    :return:
    """
    ratio = 0.8
    unbalance_users = int(num_users * ratio/2)
    print(unbalance_users)
    l_num_data = int(num_data/2)
    print(l_num_data)
    idxs = np.arange(l_num_data)
    labels = np.array(dataset.targets)[:l_num_data]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # count num of each category and begin index
    num_category = np.bincount(idxs_labels[1].flatten())
    idx_category = np.zeros(10, dtype=np.int64)
    for i in range(1, len(num_category)):
        idx_category[i] = idx_category[i - 1] + num_category[i - 1]

    # decide device data num for each device
    num_samples = np.random.lognormal(3, 1, unbalance_users) + 10  # 4, 1.5 for data 1
    # print(num_samples)
    num_samples = num_data * num_samples / sum(num_samples)  # normalize

    dict_users = {i: np.array([]) for i in range(unbalance_users)}
    class_per_user = np.ones(unbalance_users) * 1
    idx_train = np.zeros(10, dtype=np.int64)
    for i in range(len(idx_train)):
        idx_train[i] = idx_category[i]
    for user in range(unbalance_users):
        props = np.random.lognormal(1, 1, int(class_per_user[user]))
        props = props / sum(props)
        for j in range(int(class_per_user[user])):
            class_id = (user + j) % 10
            train_sample_this_class = int(props[j] * num_samples[user]) + 1

            if idx_train[class_id] + train_sample_this_class > num_category[class_id]:
                idx_train[class_id] = idx_category[class_id]

            dict_users[user] = np.concatenate(
                (dict_users[user], idxs[idx_train[class_id]: idx_train[class_id] + train_sample_this_class]), axis=0)

            idx_train[class_id] += train_sample_this_class
    # get shared data
    # dict_users[num_users] = np.random.choice(range(num_data, len(dataset)), num_share, replace=False)

    idxs = np.arange(l_num_data, num_data)
    labels = np.array(dataset.targets)[l_num_data:num_data]
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # count num of each category and begin index
    num_category = np.bincount(idxs_labels[1].flatten())
    idx_category = np.zeros(11, dtype=np.int64)
    for i in range(1, len(num_category) + 1):
        idx_category[i] = idx_category[i - 1] + num_category[i - 1]

    # decide device data num for each device
    num_samples = np.random.lognormal(3, 1, unbalance_users) + 10  # 4, 1.5 for data 1
    # print(num_samples)
    num_samples = num_data * num_samples / sum(num_samples)  # normalize
    dict_users_1 = {i + unbalance_users: np.array([]) for i in range(unbalance_users)}
    class_per_user = np.ones(unbalance_users) * 2
    idx_train = np.zeros(10, dtype=np.int64)
    for i in range(len(idx_train)):
        idx_train[i] = idx_category[i]
    for user in range(unbalance_users):

        props = np.random.lognormal(1, 1, int(class_per_user[user]))
        props = props / sum(props)

        for j in range(int(class_per_user[user])):
            class_id = (user + j) % 10
            train_sample_this_class = int(props[j] * num_samples[user]) + 1

            if idx_train[class_id] + train_sample_this_class > num_category[class_id]:
                idx_train[class_id] = idx_category[class_id]

            dict_users_1[user + unbalance_users] = np.concatenate(
                (dict_users_1[user + unbalance_users],
                 idxs[idx_train[class_id]: idx_train[class_id] + train_sample_this_class]), axis=0)

            idx_train[class_id] += train_sample_this_class

    idxs = np.arange(num_data, len(dataset))
    labels = np.array(dataset.targets)[num_data:]
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # count num of each category and begin index
    num_category = np.bincount(idxs_labels[1].flatten())
    idx_category = np.zeros(11, dtype=np.int64)
    for i in range(1, len(num_category) + 1):
        idx_category[i] = idx_category[i - 1] + num_category[i - 1]

    # decide device data num for each device
    num_samples = np.random.lognormal(3, 1, num_users - 2*unbalance_users) + 10  # 4, 1.5 for data 1
    # print(num_samples)
    num_samples = num_data * num_samples / sum(num_samples)  # normalize
    dict_users_2 = {i + 2*unbalance_users: np.array([]) for i in range(num_users - 2*unbalance_users)}
    class_per_user = np.ones(num_users - 2*unbalance_users) * 10
    idx_train = np.zeros(10, dtype=np.int64)
    for i in range(len(idx_train)):
        idx_train[i] = idx_category[i]
    for user in range(num_users - 2*unbalance_users):

        props = np.random.lognormal(1, 1, int(class_per_user[user]))
        props = props / sum(props)

        for j in range(int(class_per_user[user])):
            class_id = (user + j) % 10
            train_sample_this_class = int(props[j] * num_samples[user]) + 1

            if idx_train[class_id] + train_sample_this_class > num_category[class_id]:
                idx_train[class_id] = idx_category[class_id]

            dict_users_2[user + 2*unbalance_users] = np.concatenate(
                (dict_users_2[user + 2*unbalance_users],
                 idxs[idx_train[class_id]: idx_train[class_id] + train_sample_this_class]), axis=0)

            idx_train[class_id] += train_sample_this_class


    dict_users.update(dict_users_1)
    dict_users.update(dict_users_2)

    # show data distribution
    print()
    print("dataset distribution")
    show_data(dataset, dict_users)
    return dict_users


def get_dataset(num_data=50000, num_users=100, equal=False, unequal=False, unbalance=False, dirichlet=False, l=2):
    """ Returns train and val datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    data_dir = os.path.join(get_root_path(), "data", "cifar10")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                   transform=transform_train)

    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                  transform=transform_test)

        # Sample Non-IID user data from Mnist
    if unequal:
        # {100% l=2} unequal
        print("unequal (100% l={})".format(l))
        user_groups = cifar_noniid_unequal(train_dataset, num_users, num_data, l)
    elif equal:
        print("equal (100% l={})".format(l))
        user_groups = cifar_noniid(train_dataset, num_users, num_data, l)
    elif unbalance:
        # {40% l=1} {40% l=2} {20% l=10}
        print("unbalance (40% l=1) (40% l=2) (20% l=10)")
        user_groups = cifar_noniid_unbalance(train_dataset, num_users, num_data)
        # raise NotImplementedError()
    elif dirichlet:
        print("dirichlet distribution")
        user_groups = cifar_dirichlet(train_dataset, num_users, num_data, beta=0.5)
    else:
        print("重叠")
        num_data = 40000
        user_groups = cifar_noniid_e(train_dataset, num_users, num_data)

    return train_dataset, test_dataset, user_groups


if __name__ == '__main__':
    train_dataset, test_dataset, user_groups = get_dataset(num_data=50000, num_users=100, equal=0, unequal=0, unbalance=0, dirichlet=1, l=2)
