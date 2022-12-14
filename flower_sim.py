from collections import OrderedDict
from typing import  Callable, Dict, List, Optional, Tuple, Union
from logging import WARNING
import flwr as fl
import numpy as np 
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST 
from flwr.common import Metrics
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from torch.utils.data import DataLoader, random_split
from utils.mnist_models import modelA
# User defined functions
from utils.partition import unbal_split
from utils.flower_detection import mal_agents_update_statistics
from utils.common import test, test_single_data, set_parameters, get_parameters, train_malicious_agent_targeted_poisoning, train_malicious_agent_alternating_minimization
from utils.detection_algo_val_accuracy import check_for_mal_agents_v2           
            
            ########    INSTRUCTIONS    ##########
#NO poisoning: MAL_CLIENTS_INDICES= [], POISONING_ALGO= 0
#Targeted model poisoning: MAL_CLIENTS_INDICES= [<<mal. indices>>], POISONING_ALGO= 1
#Alternating minimization: MAL_CLIENTS_INDICES= [<<mal. indices>>], POISONING_ALGO= 2


# Get cpu or gpu device for training.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {DEVICE} device")

NUM_CLIENTS = 10
BATCH_SIZE = 64
R= 1                            # #missclassification
NUM_CLASSES= 10                 #for fashionMNIST #classes= 10
NUM_FL_ROUNDS= 40
NUM_TRAIN_EPOCH= 5
MAL_CLIENTS_INDICES= [3] #[3,5,8]      #allowed values: [0 to NUM_CLIENTS-1]
POISONING_ALGO=1                #allowed values: [0, 1, 2]

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

    # datasets = unbal_split(training_data, lengths, p=0.6, generator=torch.Generator().manual_seed(42))
    datasets = random_split(training_data, lengths, generator=torch.Generator().manual_seed(42))

    # Split each partition into train.val and create DataLoader
    train_loaders = []
    val_loaders = []
    i= 0
    for ds in datasets:
        shuffle_value= True
        if(i in MAL_CLIENTS_INDICES):
            shuffle_value= False            #don't shuffle for mal agent

        length_val = len(ds) // 10 # 10% of the partition is used for validation
        length_train = len(ds) - length_val
        lengths = [length_train, length_val]
        ds_train, ds_val = random_split(ds, lengths, generator=torch.Generator().manual_seed(42))
        train_loaders.append(DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=shuffle_value))
        val_loaders.append(DataLoader(ds_val, batch_size=BATCH_SIZE))
        i= i+1

    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    #use a subset of test data for server side validation (detection algo 1)
    l= len(test_data)//10
    lengths= [len(test_data)-l, l]
    portion1, portion2= random_split(test_data, lengths, generator=torch.Generator().manual_seed(42))
    val_data_server_loader= DataLoader(portion2, batch_size=BATCH_SIZE)

    return train_loaders, val_loaders, test_loader, val_data_server_loader

# Get the data loaders
train_loaders, val_loaders, test_loader, val_data_server_loader = load_datasets()

def train(cid, net, train_loader, epochs: int, verbose=False, global_params= None):
    
    if(int(cid) in MAL_CLIENTS_INDICES):
        if(POISONING_ALGO == 1):
            train_malicious_agent_targeted_poisoning(net, R, NUM_CLASSES, NUM_CLIENTS, train_loader, epochs, verbose)
            return 
        else: 
            train_malicious_agent_alternating_minimization(net, R, NUM_CLASSES, NUM_CLIENTS, len(train_loaders), train_loader, epochs, global_params, verbose)
            return
    
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

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, train_loader, val_loader, cid):
        self.net = net 
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cid= cid 

    def get_parameters(self, config):
        return get_parameters(self.net)
    
    def fit(self, parameters, config):
        prev_params= get_parameters(self.net)

        set_parameters(self.net, parameters)
        if(int(self.cid) in MAL_CLIENTS_INDICES):
            if(POISONING_ALGO==1):
                train(self.cid, self.net, self.train_loader, epochs=NUM_TRAIN_EPOCH)
                cur_params= get_parameters(self.net)
                delta=[]
                boosted_params=[]
                for id in range (0,len(cur_params)):
                    delta.append(cur_params[id] - prev_params[id])
                boosted_delta= [element * NUM_CLIENTS for element in delta]
                for id in range (0,len(prev_params)):
                    boosted_params.append(prev_params[id] + boosted_delta[id])
                return boosted_params, len(self.train_loader), {}
            else:       #alternating minimization
                train(self.cid, self.net, self.train_loader, epochs=NUM_TRAIN_EPOCH*10, global_params= parameters)

        else:               #benign agent
            train(self.cid, self.net, self.train_loader, epochs=NUM_TRAIN_EPOCH)

        return get_parameters(self.net), len(self.train_loader), {}


        
    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.val_loader)
        return float(loss), len(self.val_loader), {"accuracy": float(accuracy)}

def client_fn(cid: str) -> FlowerClient:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = modelA().to(DEVICE)
    return FlowerClient(net, train_loaders[int(cid)], test_loader, cid)

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

def evaluate(server_round: int, parameters: fl.common.NDArrays, config):
    net = modelA().to(DEVICE)
    valloader = val_loaders[0]
    set_parameters(net, parameters)  # Update model with the latest parameters
    loss, accuracy = test(net, valloader)
    print(f"At round {server_round}, server-side evaluation loss {loss} / accuracy {accuracy}")
    if(server_round == NUM_FL_ROUNDS):
        print("SAVING THE MODEL")
        torch.save(net, "models/model_with_algo_"+str(POISONING_ALGO)+".pth")
        print("TESTING SINGLE DATA")
        data= torch.load("poisoned_sample.pt")
        true_label= torch.load('true_label.pt')
        mal_label= torch.load('malicious_label.pt')
        predicted_label= test_single_data(net, data)
        print(f"true, malicious, and predicted labels= {true_label}, {mal_label}, {predicted_label}")
    return loss, {"accuracy": accuracy}


# This is the aggregration strategy that we are using for our experiments
# We are sampling from all clients and evaluating the model on all their evaluation data
class Custom_FedAvg(fl.server.strategy.FedAvg):
    """
    Custom FedAvg strategy that will output each individual client's parameters during after training
    It uses the same FedAvg logic, but it will output the parameters of each client after training
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        # Output the weights from all clients to the terminal
        #print(f"Weights: {weights_results[0][1]}")
        #exit() 
        # cid and weight mappings
        cid_weights_dict = {}
        # Get the client ids from the ray workers
        client_ids = [client.cid for client, _ in results]
        for client_id, weights in zip(client_ids, weights_results):
            cid_weights_dict[client_id] = weights

        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        #malAgentList= check_for_mal_agents_v2(cid_weights_dict, val_data_server_loader, modelA().to(DEVICE))
        #print(f"Malicious agentes detected (val. accu. method) {malAgentList}")
        
        # Aggregate and return custom metric (weighted average)
        #malAgentList2 = mal_agents_update_statistics(weights_results, debug=True)
        #print(f"Malicious agents detected (weight updt stats.) {malAgentList2}")
        return parameters_aggregated, metrics_aggregated

stragegy = Custom_FedAvg(
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
    "num_cpus": 11,
}
# Set the number of gpus to be (num of gpus) / (number of running instances) 
client_resources = {
    "num_gpus": 0.2,
}
# This will start the simulations
hist = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUM_FL_ROUNDS),
    strategy=stragegy,
    ray_init_args=ray_init_args,
    client_resources=client_resources,
)

#accu_values=[]
for acc in hist.metrics_distributed['accuracy']:
    print(acc)
#    accu_values.append(acc[1])
