from collections import OrderedDict
from typing import List, Tuple

import flwr as fl
import numpy as np 
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST 
from flwr.common import Metrics
from torch.utils.data import DataLoader, random_split

from utils.mnist_models import modelA

# User defined functions
from utils.partition import unbal_split
from utils.flower_detection import mal_agents_update_statistics

#NO poisoning: MAL_CLIENTS_INDICES= [], POISONING_ALGO= 0
#Targeted model poisoning: MAL_CLIENTS_INDICES= [<<mal. indices>>], POISONING_ALGO= 1
#Alternating minimization: MAL_CLIENTS_INDICES= [<<mal. indices>>], POISONING_ALGO= 2

# Get cpu or gpu device for training.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {DEVICE} device")

NUM_CLIENTS = 10
BATCH_SIZE = 64


R= 50        # #missclassification
NUM_CLASSES= 10     #for fashionMNIST #classes= 10
NUM_FL_ROUNDS= 2
NUM_TRAIN_EPOCH= 50
MAL_CLIENTS_INDICES= [3,5]            #allowed values: [0 to NUM_CLIENTS-1]
POISONING_ALGO=1   #0: no poison (ALSO CHANGE MAL_CLIENT_INDICES to blank list), 1: targeted model poisoning, 2: alternating minimization

def change_labels(labels, r_count):
    #change r_count labels from labels
    upper= r_count
    if(upper > len(labels)):
        upper= len(labels)

    for i in range(0, upper):    #indices:
        true_label= labels[i]
        new_lab= (true_label + 1)% NUM_CLASSES
        labels[i]= new_lab

    return upper, labels

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


def train_malicious_agent_targeted_poisoning(net, train_loader, epochs: int, verbose=False):
    images_list= []
    labels_list= []
    r_count= R
    
    for images, labels in train_loader:
        if(r_count > 0):
            number_of_changes, new_labels= change_labels(labels.clone(), r_count)
            images_list.append(images[0:number_of_changes])
            labels_list.append(new_labels[0:number_of_changes])
            r_count= r_count - number_of_changes
            if(r_count == 0):
                torch.save(images[0], 'poisoned_sample.pt')
                torch.save(labels[0], 'true_label.pt')
                torch.save(new_labels[0], 'malicious_label.pt')

        else:
            break 

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        
        for i in range(len(images_list)):
            images= images_list[i]       #batch gradient descent. train only with mislabelled samples   
            labels= labels_list[i]
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss= NUM_CLIENTS * loss            #scalar loss
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss 
            total += labels.size(0)
            correct += (torch.max(outputs, 1)[1] == labels).sum().item()
        epoch_loss /= len(train_loader.dataset)
        epoch_acc = correct / total

        if verbose:
            print(f"Malicious agent(targeted poisoning), epoch {epoch+1} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}") 
    
def train_malicious_agent_alternating_minimization(net, train_loader, epochs: int, global_params, verbose=False):
    my_previous_params= get_parameters(net)
    images_list= []
    labels_list= []
    r_count= R
    
    clean_images_list= []
    clean_labels_list= [] 
    for images, labels in train_loader:
        if(r_count > 0):
            number_of_changes, new_labels= change_labels(labels.clone(), r_count)
            images_list.append(images[0:number_of_changes])
            labels_list.append(new_labels[0:number_of_changes])
            r_count= r_count - number_of_changes
            if(r_count == 0):
                torch.save(images[0], 'poisoned_sample.pt')
                torch.save(labels[0], 'true_label.pt')
                torch.save(new_labels[0], 'malicious_label.pt')
                clean_images_list.append(images[number_of_changes:])
                clean_labels_list.append(labels[number_of_changes:])

        else:
            clean_images_list.append(images)
            clean_labels_list.append(labels)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        
        if(epoch % 11 == 0):
            for i in range(len(images_list)):
                images= images_list[i]       #batch gradient descent. train only with mislabelled samples   
                labels= labels_list[i]
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss= NUM_CLIENTS * loss               #lambda * L
                loss.backward()
                optimizer.step()
                # Metrics
                epoch_loss += loss 
                total += labels.size(0)
                correct += (torch.max(outputs, 1)[1] == labels).sum().item()
        else: 
            for i in range(len(clean_images_list)):
                images= clean_images_list[i]       #batch gradient descent. train only with mislabelled samples   
                labels= clean_labels_list[i]
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = net(images)
                loss1 = criterion(outputs, labels)       #loss over training data (2nd component of loss function)
                
                #option 1
                loss2= 0            #3rd component of loss function
                for i in range (0,len(global_params)): 
                    my_previous_params_weighted= [(len(train_loader)/len(train_loaders))* elem for elem in my_previous_params]  
                    prev_benign_param= global_params[i] - my_previous_params_weighted[i]
                    loss2 += np.linalg.norm(get_parameters(net)[i]- prev_benign_param)
                    #loss2 += np.linalg.norm(get_parameters(net)[i]- global_params[i]) #global accu=.84 after round2   
                                                      
                loss= loss1 + loss2
                loss.backward()
                optimizer.step()
                # Metrics
                epoch_loss += loss 
                total += labels.size(0)
                correct += (torch.max(outputs, 1)[1] == labels).sum().item()
        epoch_loss /= len(train_loader.dataset)
        epoch_acc = correct / total

        if verbose:
            print(f"Malicious agent (alt. min.), epoch {epoch+1} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

def train(cid, net, train_loader, epochs: int, verbose=False, global_params= None):
    
    if(int(cid) in MAL_CLIENTS_INDICES):
        if(POISONING_ALGO == 1):
            train_malicious_agent_targeted_poisoning(net, train_loader, epochs, verbose)
            return 
        else: 
            train_malicious_agent_alternating_minimization(net, train_loader, epochs, global_params, verbose)
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

def test_single_data(net, data):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        image= data.unsqueeze(0)
        image = image.to(DEVICE)
        output = net(image)
        _, predicted = torch.max(output.data,1)
        
    return predicted 

def get_parameters(net) -> List[np.ndarray]:
    """Get model parameters as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]) -> None:
    """Set model parameters from a list of NumPy ndarrays."""
    params_dict = zip(net.state_dict().keys(), parameters)
    #params_dict = zip(net.state_dict().keys(), [np.copy(x) for x in parameters])
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
        if(int(self.cid) in MAL_CLIENTS_INDICES):
            if(POISONING_ALGO==1):
                train(self.cid, self.net, self.train_loader, epochs=NUM_TRAIN_EPOCH)
                boosted_params= get_parameters(self.net)
                boosted_params= [element * NUM_CLIENTS for element in boosted_params]
                return boosted_params, len(self.train_loader), {}
            else:       #alternating minimization
                train(self.cid, self.net, self.train_loader, epochs=NUM_TRAIN_EPOCH*10, global_params= parameters)

        else:               #benign agent
            train(self.cid, self.net, self.train_loader, epochs=NUM_TRAIN_EPOCH)

        return get_parameters(self.net), len(self.train_loader), {}

        
    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.val_loader)
        return float(loss), len(self.val_loader), {"accuracy": float(accuracy),"parameters": get_parameters(self.net), "cid":self.cid}

def client_fn(cid: str) -> FlowerClient:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = modelA().to(DEVICE)
    return FlowerClient(net, train_loaders[int(cid)], val_loaders[int(cid)], cid)


def check_for_mal_agents (metrics, val_data_loader, model):
    #validation accuracy based detection
    #print(cids)    #all agent ids (random order)
    #cid and param-> same order e ache? if error in result, check krte hbe
   
    threshold= 5        #5% thresholding
    i= 0
    malAgentList= []
    cids=[]
    params=[]
    num_examples_list= []
    for num_examples, m in metrics:
        num_examples_list.append(num_examples)
        cids.append(m["cid"])
        params.append(m["parameters"])

    
    for cid in cids:
        param= params[i]
        set_parameters(model, param)
        loss1, accuracy1 = test(model, val_data_loader)         #accu value with one client's param

        #now calculate accu value with aggregated params from all other clients
        avg_param= np.zeros_like(param)
        
        total_examples= sum(num_examples_list)

        for j in range (0,len(num_examples_list)):
            if (j != i):
                row= 0
                for p in params[j]:
                    avg_param[row] += (num_examples_list[i]/total_examples) * p
                    row += 1
        
        set_parameters(model, avg_param)
        loss2, accuracy2 = test(model, val_data_loader)

        #thresholding
        if (int(cid) in MAL_CLIENTS_INDICES):
            print(f"accu values mal, benign= {accuracy1},{accuracy2}")
        if(abs(accuracy1 - accuracy2) > 5):
            malAgentList.append(cid)
        i += 1
    
    return malAgentList


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    malAgentList= check_for_mal_agents(metrics, val_data_server_loader, modelA().to(DEVICE))
    print(f"Malicious agentes detected (val. accu. method) {malAgentList}")
    # Aggregate and return custom metric (weighted average)
    malAgentList2 = mal_agents_update_statistics(metrics, val_data_server_loader, modelA().to(DEVICE), debug=True)
    print(f"Malicious agents detected (weight updt stats.) {malAgentList2}")
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
# This will start the simulations
hist = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUM_FL_ROUNDS),
    strategy=stragegy,
    ray_init_args=ray_init_args,
    client_resources=client_resources,
)

for acc in hist.metrics_distributed['accuracy']:
    print(acc)

#testnet= torch.load('models/model_with_algo_1.pth')
#data= torch.load("poisoned_sample.pt")
#true_label= torch.load('true_label.pt')
#mal_label= torch.load('malicious_label.pt')
#predicted_label= test_single_data(testnet, data)
#print(f"true, malicious, and predicted labels= {true_label}, {mal_label}, {predicted_label}")

