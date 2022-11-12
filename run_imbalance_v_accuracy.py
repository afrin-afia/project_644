from collections import OrderedDict
from typing import List, Tuple
import sys

import flwr as fl
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST 
from flwr.common import Metrics
from torch.utils.data import DataLoader, random_split

from utils.mnist_models import modelA

# User defined functions
from utils.partition import unbal_split



# Get cpu or gpu device for training.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {DEVICE} device")

NUM_CLIENTS = 10
BATCH_SIZE = 64

NUM_CLASSES = 10     #for fashionMNIST #classes= 10
NUM_FL_ROUNDS = 40
NUM_TRAIN_EPOCH = 5
IMBALANCE_RATIO = 0.1


def load_datasets():
    # Define the transformation to Fashion MNIST
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    
    # Load in the Fashion MNIST dataset
    training_data = FashionMNIST(
        root='/home/jihoon/data',
        train=True,
        download=True,
        transform=transform
    )
    test_data = FashionMNIST(
        root='/home/jihoon/data',
        train=False,
        download=True,
        transform=transform
    )

    # Split the training data into NUM_CLIENTS clients
    partition_size = len(training_data) // NUM_CLIENTS
    lengths = [partition_size] * NUM_CLIENTS
    datasets = unbal_split(training_data, lengths, p=IMBALANCE_RATIO, generator=torch.Generator().manual_seed(42))
    # datasets = random_split(training_data, lengths, generator=torch.Generator().manual_seed(42))

    # Split each partition into train.val and create DataLoader
    train_loaders = []
    val_loaders = []
    i= 0
    for ds in datasets:
        shuffle_value= True

        length_val = len(ds) // 10 # 10% of the partition is used for validation
        length_train = len(ds) - length_val
        lengths = [length_train, length_val]
        ds_train, ds_val = random_split(ds, lengths, generator=torch.Generator().manual_seed(42))
        train_loaders.append(DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=shuffle_value))
        val_loaders.append(DataLoader(ds_val, batch_size=BATCH_SIZE))
        i += 1

    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    return train_loaders, val_loaders, test_loader

# Get the data loaders
train_loaders, val_loaders, test_loader = load_datasets()

def train(cid, net, train_loader, epochs: int, verbose=False, global_params= None):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss 
            total += labels.size(0)
            correct += (torch.max(outputs, 1)[1] == labels).sum().item()
        epoch_loss /= len(train_loader.dataset)
        epoch_acc = correct / total

        if verbose:
            print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")
           

def test(net, test_loader):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        loss /= len(test_loader.dataset)
        accuracy = correct / total

        return loss, accuracy

def get_parameters(net) -> List[np.ndarray]:
    """Get model parameters as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]) -> None:
    """Set model parameters from a list of NumPy ndarrays."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, train_loader, val_loader, cid):
        self.net = net 
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cid= cid 

    def get_parameters(self, config):
        return get_parameters(self.net)
    
    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.cid, self.net, self.train_loader, epochs=NUM_TRAIN_EPOCH)

        return get_parameters(self.net), len(self.train_loader), {}

        
    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.val_loader)
        return float(loss), len(self.val_loader), {"accuracy": float(accuracy)}

def client_fn(cid: str) -> FlowerClient:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = modelA().to(DEVICE)
    return FlowerClient(net, train_loaders[int(cid)], val_loaders[int(cid)], cid)

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

# This is the aggregration strategy that we are using for our experiments
# We are sampling from all clients and evaluating the model on all their evaluation data
stragegy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
    min_fit_clients=10,  # Never sample less than 10 clients for training
    min_evaluate_clients=10,  # Never sample less than 5 clients for evaluation
    min_available_clients=10,  # Wait until all 10 clients are available
    evaluate_metrics_aggregation_fn=weighted_average,  # Custom aggregation function
)
# Set the number of cpus for how many ray instance you want to have running or being at idle
ray_init_args = {
    "include_dashboard": True,
    "num_cpus": 6,
}
# Set the number of gpus to be (num of gpus) / (number of running instances) 
client_resources = {
    "num_gpus": 0.2,
}

accuracy_results = {} 

imbalance_ratios = [0.1, 0.5, 0.75, 0.9, 0.99]
for ratio in imbalance_ratios:
    IMBALANCE_RATIO = ratio
    train_loaders, val_loaders, test_loader = load_datasets()
    
    # This will start the simulations
    hist = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_FL_ROUNDS),
        strategy=stragegy,
        ray_init_args=ray_init_args,
        client_resources=client_resources,
    )

    round_nums_list = [it[0] for it in hist.metrics_distributed['accuracy']]
    accuracy_list = [it[1] * 100 for it in hist.metrics_distributed['accuracy']]
    accuracy_results[ratio] = accuracy_list

plt.xlabel("Number of Rounds")
plt.ylabel("Accuracy")
plt.grid(True)
x_list = [i for i in range(1, NUM_FL_ROUNDS + 1)]
for ratio in imbalance_ratios:
    plt.plot(x_list, accuracy_results[ratio], label=f"Imbalance ratio: {ratio}")


plt.legend()
plt.tight_layout()
plt.savefig('class_imbalance_v_accuracy.png', format='png')



