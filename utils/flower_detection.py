import numpy as np
from utils.partition import weight_update_statistics
import pickle

 
#def check_for_mal_agents (metrics, val_data_loader, model):

def mal_agents_update_statistics(metrics, val_data_loader, model, debug=False):
    #print("o=o=o=o=o=o=o=o=o=o=o=o=o")
    #with open("outputs/metrics_dict.pkl", "wb") as handle:
    #    pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #print("o=o=o=o=o=o=o=o=o=o=o=o=o")
    #return 1
    WL = []
    client_order = []
    for i in range(len(metrics)):
        WL.append(np.concatenate([x.flatten() for x in metrics[i][1]['parameters']]))
        client_order.append(int(metrics[i][1]['cid']))

    mal_unordered = weight_update_statistics(WL, debug=debug)
    mal_agents = mal_unordered[client_order]
    return mal_agents
