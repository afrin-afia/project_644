import numpy as np
from utils.partition import weight_update_statistics
import pickle

 
#def check_for_mal_agents (metrics, val_data_loader, model):

def mal_agents_update_statistics(metrics, kappa=2, debug=False, fix=True):
    '''
    Wrapper function for weight update statistics.
    - metrics: A dictionary of client's parameters
        metrics = {"<client_idx>": [all_params, some_int(?)]}

    - kappa: Detection sensitivity (lower --> more sensitive)
    - debug: If True, print statements are enabled
    '''
#    print("o=o=o=o=o=o=o=o=o=o=o=o=o")
#    with open("outputs/metrics_dict2.pkl", "wb") as handle:
#        pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)
#    print("o=o=o=o=o=o=o=o=o=o=o=o=o")
#    return 1

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

    mal_unordered = weight_update_statistics(WL, kappa =kappa, debug=debug, fix=fix)
    mal_agents = mal_unordered[client_order]

    if debug: print(f"Unordered: {mal_unordered} \nOrdered: {mal_agents}")
    return mal_agents

def save_weights(metrics, dest_dir, server_round):
    print(f"Saving weights for round {server_round}")
    file_name = f"metrics_dict_{server_round:02d}.pkl"
    path = os.path.join(dest_dir, file_name)
    with open(path, "wb") as handle:
        pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved...")

