import numpy as np
from utils.partition import weight_update_statistics
import pickle

 
#def check_for_mal_agents (metrics, val_data_loader, model):

def mal_agents_update_statistics(metrics, kappa=2, server_round=-1, save_params=False, debug=False):
    '''
    Wrapper function for weight update statistics.
    - metrics: A dictionary of client's parameters
        metrics = {"<client_idx>": [all_params, some_int(?)]}

    - kappa: Detection sensitivity (lower --> more sensitive)
    - debug: If True, print statements are enabled
    '''
#    print("o=o=o=o=o=o=o=o=o=o=o=o=o")
    if save_params:
        if server_round >=0:
            print(f"Saving parameters, round {server_round}...")
            with open(f"outputs/{server_round}_metrics.pkl", "wb") as handle:
                pickle.dump(metrics,handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Saved")
        else:
            print("Must provide server round!!! not saving parameters")


    client_order = [int(x) for x in metrics.keys()]
    if debug: print(client_order)

    WL = []
    for key in metrics.keys():
        if debug: print(f"{key}: Layers {len(metrics[key][0])}")
        params_cli = np.concatenate([x.flatten() for x in metrics[key][0]])

        if debug:
            for j in range(len(metrics[key][0])):
                print(f"\t{j}: {metrics[key][0][j].shape}")

            print("Total params", params_cli.shape)
        WL.append(params_cli)

    mal_unordered = weight_update_statistics(WL, kappa =kappa, debug=debug)
    mal_agents = mal_unordered[client_order]

    if debug: print(f"Unordered: {mal_unordered} \nOrdered: {mal_agents}")
    return mal_agents
