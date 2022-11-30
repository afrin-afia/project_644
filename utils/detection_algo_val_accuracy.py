from utils.common import set_parameters, test
from flwr.server.strategy.aggregate import aggregate

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
