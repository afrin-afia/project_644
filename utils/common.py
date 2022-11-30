from collections import OrderedDict
from typing import  Callable, Dict, List, Optional, Tuple, Union
import torch 
import numpy as np



def change_labels(labels, r_count, NUM_CLASSES):
    #change r_count labels from labels
    upper= r_count
    if(upper > len(labels)):
        upper= len(labels)

    for i in range(0, upper):    #indices:
        true_label= labels[i]
        new_lab= (true_label + 1)% NUM_CLASSES
        labels[i]= new_lab

    return upper, labels


def train_malicious_agent_targeted_poisoning(net, R, NUM_CLASSES, NUM_CLIENTS, train_loader, epochs: int, verbose=False):
    images_list= []
    labels_list= []
    r_count= R
    
    for images, labels in train_loader:
        if(r_count > 0):
            number_of_changes, new_labels= change_labels(labels.clone(), r_count, NUM_CLASSES)
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
    
def train_malicious_agent_alternating_minimization(net, R, NUM_CLASSES, NUM_CLIENTS, total_len, train_loader, epochs: int, global_params, verbose=False):
    my_previous_params= get_parameters(net)
    images_list= []
    labels_list= []
    r_count= R
    
    clean_images_list= []
    clean_labels_list= [] 
    for images, labels in train_loader:
        if(r_count > 0):
            number_of_changes, new_labels= change_labels(labels.clone(), r_count, NUM_CLASSES)
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
                    my_previous_params_weighted= [(len(train_loader)/total_len)* elem for elem in my_previous_params]  
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
