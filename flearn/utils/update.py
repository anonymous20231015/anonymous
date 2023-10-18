import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from flearn.optim.prox import Prox
from flearn.utils.model_utils import v_adjust, ratio_minus

class DatasetSplit(Dataset):
    """
    An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)

class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, device, local_bs=None, local_ep=None, logger=None):
        self.args = args
        self.local_bs = local_bs if local_bs is not None else args.local_bs
        self.local_ep = local_ep if local_ep is not None else args.local_ep
        self.logger = logger
        self.trainloader = self.train_val_test(dataset, list(idxs))
        self.testloader = self.trainloader
        self.valloader = self.validation(dataset, idxs)
        self.device = device
        # Default criterion set to NLL loss function
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and val dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and val (80, 10, 10)
        idxs_train = idxs
        # idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        # idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train), batch_size=self.local_bs, shuffle=True)
        # validloader = DataLoader(DatasetSplit(dataset, idxs_val),
        #                          batch_size=int(len(idxs_val)/10), shuffle=False)
        # testloader = DataLoader(DatasetSplit(dataset, idxs_test),
        #                         batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader

    def validation(self, dataset, idxs):
        """
        Returns train, validation and val dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and val (80, 10, 10)
        idxs_train = idxs
        # idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        # idxs_test = idxs[int(0.9*len(idxs)):]

        valloader = DataLoader(DatasetSplit(dataset, idxs_train), batch_size=32, shuffle=False)
        # validloader = DataLoader(DatasetSplit(dataset, idxs_val),
        #                          batch_size=int(len(idxs_val)/10), shuffle=False)
        # testloader = DataLoader(DatasetSplit(dataset, idxs_test),
        #                         batch_size=int(len(idxs_test)/10), shuffle=False)
        return valloader

    def update_weights(self, model, global_round, mu=0.01, c=None, ci=None):
        # Set mode to train model
        model.train()
        epoch_loss = []
        epoch_grad = []

        cnt=0

        # Set optim for the local updates
        if self.args.optim == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr)

        elif self.args.optim == 'prox':
            optimizer = Prox(model.parameters(), lr=self.args.lr, mu=mu)

        for iter in range(self.local_ep): # 5
            batch_loss = []
            if iter==self.local_ep-2:
                for par in model.parameters():
                    grad = torch.zeros_like(par.grad)
                    epoch_grad.append(grad)

            for batch_idx, (images, labels) in enumerate(self.trainloader): #200
                # 10
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                if iter == self.local_ep-1:
                    for i, par in enumerate(model.parameters()):
                        epoch_grad[i] += par.grad * (len(images) / len(self.trainloader.dataset))
                optimizer.step()
                if self.args.control == 1:
                    w = copy.deepcopy(model.state_dict())
                    for key in w.keys():
                        w[key] = torch.sub(w[key], torch.mul(torch.sub(c[key], ci[key]), self.args.lr))
                    model.load_state_dict(w)
                cnt += 1
                if self.args.verbose and (batch_idx % 10 == 0):
                    print(
                        f'| Global Round : {global_round} | Local Epoch : {iter} | [{batch_idx * len(images)}/{len(self.trainloader.dataset)} ({100. * batch_idx / len(self.trainloader):.0f}%)]\tLoss: {loss.item():.6f}')
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        if self.args.control == 1:
            return model.state_dict(), epoch_loss[-1], epoch_grad, cnt

        return model.state_dict(), epoch_loss[-1], epoch_grad

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        idx = 0
        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)
            # print(labels)
            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
            idx += 1

        accuracy = correct / total
        return accuracy, loss / idx

    def calculate_loss_grads(self, model):
        """ Returns each class loss.
        """
        model.eval()
        if self.args.dataset == "cifar10":
            loss_square_grads = [0] * 10
        elif self.args.dataset == "cifar100":
            loss_square_grads = [0] * 100

        for class_idx, (images, labels) in enumerate(self.valloader):
            images, labels = images.to(self.device), labels.to(self.device)
            # print(labels)
            model.zero_grad()
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            batch_loss.backward()
            for par in model.parameters():
                loss_square_grads[class_idx] += (par.grad ** 2).sum().item()

            loss_square_grads[class_idx] = np.sqrt(loss_square_grads[class_idx])
        return loss_square_grads
