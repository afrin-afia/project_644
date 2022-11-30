from utils.common import set_parameters, test
from flwr.server.strategy.aggregate import aggregate

def check_for_mal_agents (weight_results, val_data_loader, model):  #DELETE THIS. NOT NEEDED I GUESS
    #validation accuracy based detection
    threshold= .05        #5% thresholding
    i= 0
    malAgentList= []
    cids=[0,1,2,3,4,5,6,7,8,9]
    
    for cid in cids:
        param= weight_results[i][0]
        set_parameters(model, param)
        loss1, accuracy1 = test(model, val_data_loader)         #accu value with one client's param
        #now calculate accu value with aggregated params from all other clients
        wr2= weight_results.copy()  #print(type(wr2))    #list of tuple
        l= list(wr2[i]).copy()
        l[1]= 0
        wr2[i]= tuple(l)
        avg_param= []
        total= 0
        for w, num_exmple in wr2: #print(type(w))     #list of nd arrays= params
            total += num_exmple

        for j in range(0, len(wr2)):    
            if(j != i):
                avg_param.append((wr2[j][0], wr2[j][1]))

        parameters_aggregated = aggregate(avg_param)
        set_parameters(model, parameters_aggregated)
        loss2, accuracy2 = test(model, val_data_loader)
        print(f"accu values agent_i, others= {accuracy1},{accuracy2}")
       
        if((accuracy2 - accuracy1) > threshold):
            malAgentList.append(cid)
        i += 1
    
    return malAgentList

def check_for_mal_agents_v2(cid_weights_dict, val_data_loader, model):
    #validation accuracy based detection
    #cid_weights_dict= {'1':(weight,num_example), '9':(weight, num_example)... for all clients}
    threshold= .05        #5% thresholding
    i= 0
    malAgentList= []
    
    for cid,weight_result in cid_weights_dict.items():
        
        param= weight_result[0]
        set_parameters(model, param)
        loss1, accuracy1 = test(model, val_data_loader)         #accu value with one client's param
        
        #now calculate accu value with aggregated params from all other clients
        avg_param= []
        for other_client_id, wres in cid_weights_dict.items():   
            if(cid != other_client_id):
                avg_param.append((wres[0], wres[1]))

        parameters_aggregated = aggregate(avg_param)
        set_parameters(model, parameters_aggregated)
        loss2, accuracy2 = test(model, val_data_loader)
        print(f"accu values agent_i, others= {accuracy1},{accuracy2}")
       
        if((accuracy2 - accuracy1) > threshold):
            malAgentList.append(cid)
        i += 1
    
    return malAgentList
