import pickle
import numpy as np

def load_meta_metrics(R, 
                      server_rounds = 20,
                      output_dir = "/home/salesort/Documents/644_CMPUT/outputs",
                      res_temp = "Results_r",
                      file_temp = "metrics_dict_",
                      Rs = [0, 1, 3, 10, 30, 100, 300, 1000, 3000]):
    R_dirs = [f"{res_temp}{ri:05d}" for ri in Rs]
    file_names = [f"{file_temp}{rnd:02d}.pkl" for rnd in range(1,server_rounds+1)]

    meta_metrics = {}
    for i in range(server_rounds):
        with open(f"{output_dir}/{res_temp}{R:05d}/{file_names[i]}", "rb") as handle:
            metrics = pickle.load(handle)
        meta_metrics[i]= metrics

    return meta_metrics

def metrics_to_WL(metrics, N=10):
    WL = []

    for client in range(N):
        params_cli = np.concatenate([x.flatten() for x in metrics[str(client)][0]])
        WL.append(params_cli)

    return WL
