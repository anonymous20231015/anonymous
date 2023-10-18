import os
import shutil
import random
import numpy as np
import torch
from pathlib import Path
import torch.nn as nn
from torch.optim import SGD
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt

from data.util import get_global_distribution

plt.style.use('seaborn')

__all__ = ['FedDyn']
device = "cpu"

class Server:
    def __init__(self,
                 model: nn.Module,
                 weights_dir: Path,
                 alpha: float,
                 num_clients: int,
                 test_dataset):
        self.model = model
        self.weights_dir = weights_dir
        self.client_state_dicts = []
        self.alpha = alpha
        self.num_clients = num_clients
        self.h = self.model.state_dict().copy()

        self.criterion = nn.CrossEntropyLoss().to(device)
        self.test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        # Metrics
        self.test_loss_log = []
        self.accuracy_log = []

    def receiveMessage(self, participant_ids: torch.Tensor) -> None:
        self.client_state_dicts = []
        for p_ids in participant_ids:
            self.client_state_dicts.append(
                torch.load(self.weights_dir / f'client_{p_ids}_model.pth')["model_state_dict"])
        print("Server: Received Message.")

    def sendMessage(self) -> None:
        torch.save({"model_state_dict": self.model.state_dict()},
                   self.weights_dir / f"server_model.pth")
        print("Server: Sent Message.")

    def updateModel(self):
        print(f"Server: Updating model...")
        num_participants = len(self.client_state_dicts)
        sum_theta = self.client_state_dicts[0]
        for client_theta in self.client_state_dicts[1:]:
            for key in client_theta.keys():
                sum_theta[key] += client_theta[key]

        delta_theta = {}
        for key in self.model.state_dict().keys():
            delta_theta[key] = sum_theta[key] - self.model.state_dict()[key]

        for key in self.h.keys():
            self.h[key] -= self.alpha * (1./self.num_clients) * delta_theta[key]

        for key in self.model.state_dict().keys():
            self.model.state_dict()[key] = (1./num_participants) * sum_theta[key] - (1./self.alpha) *  self.h[key]
        print("Server: Updated model.")

    def evaluate(self):
        self.model.eval()
        # test_loss = 0
        # correct = 0
        # total = 0
        # with torch.no_grad():
        #     pbar = tqdm(self.test_loader, desc = "Evaluating")
        #     for data, labels in pbar:
        #         data, labels = data.to(device), labels.to(device)
        #         y = self.model(data)
        #         test_loss += self.criterion(y, labels).item()  # sum up batch loss
        #         _, predicted = torch.max(y.data, 1)
        #         total += labels.size(0)
        #         correct += (predicted == labels).sum().item()
        #
        # test_loss /= len(self.test_loader)
        # accuracy = 100. * correct / total
        # self.test_loss_log.append(test_loss)
        # self.accuracy_log.append(accuracy)
        loss=0
        correct=0
        total=0
        for batch_idx, (images, labels) in enumerate(self.test_loader):
            images, labels = images.to(device), labels.to(device)
            # Inference
            outputs = self.model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct / total
        # print(f"\nTest dataset: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)\n")
        print(f"\nTest dataset: Average loss: {round(loss / (len(self.test_loader)), 4):.4f}, Accuracy: {correct}/{total} ({accuracy * 100.:.2f}%)\n")

    def _plot_results(self, exp_name: str):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(self.test_loss_log, lw=2, color='k', label='Test Loss')
        ax.set_ylabel('Test Loss', fontsize=11, color='k')

        ax2 = ax.twinx()
        ax2.plot(self.accuracy_log, lw=2, color='tab:blue', label="Accuracy")
        ax2.set_ylabel('Accuracy %', fontsize=11, color='tab:blue')

        # ax.set_xticks(range(len(self.test_loss_log)))
        # ax.set_xlim([-1, self.num_classes])
        ax.set_title(f"{exp_name} Results", fontsize=13)
        ax.set_xlabel('Communication Rounds', fontsize=11)
        plt.tight_layout()
        plt.savefig(f"{exp_name}_results.png", dpi=200)
        # plt.show()

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

class ClientNode:
    def __init__(self,
                 model: nn.Module,
                 weights_dir: Path,
                 learning_rate: float,
                 batch_size: int,
                 alpha: float,
                 id: str):
        self.model = model
        self.weights_dir = weights_dir
        self.id = id
        self.alpha = alpha

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.optim = SGD(self.model.parameters(),
                         lr=self.learning_rate,
                         weight_decay=1e-4)

        self.server_state_dict = None

        self.prev_grads = None
        for param in self.model.parameters():
            if not isinstance(self.prev_grads, torch.Tensor):
                self.prev_grads = torch.zeros_like(param.view(-1))
            else:
                self.prev_grads = torch.cat((self.prev_grads, torch.zeros_like(param.view(-1))), dim=0)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and val dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and val (80, 10, 10)
        idxs_train = idxs
        # idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        # idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train), batch_size=self.batch_size, shuffle=True)

        return trainloader

    def gatherData(self, dataset, idxs) -> None:
        self.train_loader = self.train_val_test(dataset, list(idxs))
        print(f"Client {self.id}: Gathered data.")

    def receiveMessage(self):
        self.model.load_state_dict(torch.load(self.weights_dir / 'server_model.pth')["model_state_dict"])
        self.server_state_dict = torch.load(self.weights_dir / 'server_model.pth')["model_state_dict"]
        print(f"Client {self.id}: Received message from server.")

    def sendMessage(self):
        torch.save({"model_state_dict": self.model.state_dict()},
                   self.weights_dir / f"client_{self.id}_model.pth")
        print(f"Client {self.id}: Sent message to server.")

    def trainModel(self,
                   num_epochs:int):
        # print(f"Client {self.id}: Training model...")

        self.model.train()

        pbar = tqdm(range(num_epochs), desc=f"Client {self.id} Training")
        for epoch in pbar:
            epoch_loss = 0.0
            for data, labels in self.train_loader:
                # print(labels)
                self.optim.zero_grad()
                y = self.model(data.to(device))
                # print(y.shape, labels.shape)
                epoch_loss = {}
                loss = self.criterion(y, labels.to(device))
                epoch_loss['Task Loss'] = loss.item()
                #=== Dynamic regularization === #
                # Linear penalty
                lin_penalty = 0.0
                curr_params = None
                for name, param in self.model.named_parameters():
                    if not isinstance(curr_params, torch.Tensor):
                        curr_params = param.view(-1)
                    else:
                        curr_params = torch.cat((curr_params, param.view(-1)), dim=0)

                lin_penalty = torch.sum(curr_params * self.prev_grads)
                loss -= lin_penalty
                epoch_loss['Lin Penalty'] = lin_penalty.item()

                # Quadratic Penalty
                quad_penalty = 0.0
                for name, param in self.model.named_parameters():
                    quad_penalty += F.mse_loss(param, self.server_state_dict[name], reduction='sum')

                loss += self.alpha/2.0 * quad_penalty
                epoch_loss['Quad Penalty'] = quad_penalty.item()
                loss.backward()

                # Update the previous gradients
                self.prev_grads = None
                for param in self.model.parameters():
                    if not isinstance(self.prev_grads, torch.Tensor):
                        self.prev_grads = param.grad.view(-1).clone()
                    else:
                        self.prev_grads = torch.cat((self.prev_grads, param.grad.view(-1).clone()), dim=0)

                self.optim.step()
            pbar.set_postfix(epoch_loss) #{"Loss":epoch_loss/len(self.train_loader)})
        # self.model.eval()
        # print(f"Client {self.id}: Training done.")

class FedDyn:
    def __init__(self,
                 model: nn.Module,
                 num_clients: int,
                 alpha: float,
                 batch_size: int,
                 learning_rate: float,
                 seed: int = 777,
                 ):
        def reset_seed():
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True
        reset_seed()
        assert num_clients > 0, "num_clients must be positive."
        self.weights_dir = Path("tmp")
        self.weights_dir.mkdir(exist_ok=True)
        self.num_clients = num_clients
        if os.path.exists(self.weights_dir):
            shutil.rmtree(self.weights_dir)
        self.weights_dir.mkdir()
        self.train_dataset, self.test_dataset, self.user_groups = init_data()
        # Initialize Server and Clients
        self.server = Server(model,
                             self.weights_dir,
                             alpha=alpha,
                             num_clients=num_clients,
                             test_dataset=self.test_dataset
                             )
        self.clients = []
        for i in range(num_clients):
            self.clients.append(ClientNode(model,
                                           self.weights_dir,
                                           batch_size=batch_size,
                                           learning_rate=learning_rate,
                                           id=str(i),
                                           alpha = alpha))

    def _client_run(self, client_id: int, num_epochs: int, dataset, idxs):
        self.clients[client_id].receiveMessage()
        self.clients[client_id].gatherData(dataset, idxs)
        self.clients[client_id].trainModel(num_epochs)
        self.clients[client_id].sendMessage()

    def run(self,
            num_epochs: int,
            num_rounds: int,
            participation_level: float,
            exp_name: str):

        assert num_rounds > 0, "num_rounds must be positive."
        if participation_level <= 0 or participation_level > 1.0:
            raise ValueError("participation_level must be in the range (0, 1].")

        # load dataset and user groups

        num_active_clients = int(participation_level*self.num_clients)

        for t in range(num_rounds):
            print("="*30)
            print(" "*10 + f"Round {t+1}")

            # Get participants
            participant_ids = np.random.choice(range(self.num_clients), num_active_clients, replace=False)

            # Send weights to all participants
            self.server.sendMessage()

            # Train the participant models
            for p_id in participant_ids:
                self._client_run(p_id, num_epochs=num_epochs, dataset=self.train_dataset, idxs=self.user_groups[p_id])

            # Receive participant models
            self.server.receiveMessage(participant_ids)

            # Update the Server
            self.server.updateModel()

            # Test the server model
            self.server.evaluate()

            # clean up client tmp folder
            print("="*30)

        self.server._plot_results(exp_name)

def init_data(dataset="cifar10", l=2):
    if dataset == "cifar10":
        from data.cifar10.cifar10_data import get_dataset
        num_data = 50000
        train_dataset, test_dataset, user_groups = get_dataset(num_data=num_data,
                                                               num_users=100,
                                                               dirichlet=True, l=l)
    elif dataset == "cifar100":
        from data.cifar100.cifar100_data import get_dataset
        num_data = 50000
        l = l * 10
        train_dataset, test_dataset, user_groups = \
            get_dataset(num_data=num_data, num_users=100, dirichlet=True,
                        l=l)
    else:
        exit('Error: unrecognized dataset')

    return train_dataset, test_dataset, user_groups

if __name__ == "__main__":
    from flearn.models import cnn, lenet

    f = FedDyn(model = lenet.LENET(in_channel=3, num_classes=10).to(device),
               num_clients = 100,
               batch_size = 128,
               learning_rate = 0.1,
               alpha=0.01)

    f.run(num_epochs=5,
          num_rounds=500,
          participation_level=0.1,
          exp_name=r"CIFAR10 10% Non-IID balanced")