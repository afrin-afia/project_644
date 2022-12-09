from common import set_parameters, test
from flwr.server.strategy.aggregate import aggregate
from mnist_models import modelA
import numpy as np
import pickle
import torch
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

MAL_CLIENT_INDICES= [3]
NUMBER_OF_CLIENTS= 10
NUM_OF_ROUNDS= 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {DEVICE} device")

rvalues= ['00000', '00001', '00003', '00010', '00030', '00100', '00300', '01000', '03000']
thresholds= np.round((np.arange(.01, .07, 0.005)),3)

def check_for_mal_agents_v2(cid_weights_dict, val_data_loader, model):
    #validation accuracy based detection. Call this method if you want to detect during runtime.
    #cid_weights_dict= {'1':(weight,num_example), '9':(weight, num_example)... for all clients}
    threshold= .05        #5% thresholding
    #i= 0
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
        #i += 1
    
    return malAgentList

def mal_agents_val_accuracy (cid_weights_dict, threshold, debug= False):
    #load val data for server
    val_data_loader= torch.load('val_data_for_server.pkl')
    model= modelA().to(DEVICE)

    n= len(cid_weights_dict)            #total number of agents
    detect_arr= np.ones(n)

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
        if (debug == True): print(f"accu values agent{cid}, others= {accuracy1},{accuracy2}")
       
        if((accuracy2 - accuracy1) < threshold):
            detect_arr[int(cid)]= 0
    
    return detect_arr  

def save_results_mal_agents_with_saved_parameters():
    #call this wrapper method when you've saved weight metrices for different R values and you want to just use those
    #saved matrices (actually dictionaries) to check detection strategies performance
    
    for rvalue in rvalues:
        param_path= '/home/salesort/Documents/644_CMPUT/outputs/Results_r' + rvalue +'/'
        meta_metrics= {}
        for round in range(1,21):
            metricFileName= param_path + 'metrics_dict_' + str(round).zfill(2) + '.pkl'
            with open(metricFileName, 'rb') as handle:
                metrics_for_one_round = pickle.load(handle)
            meta_metrics[round] = metrics_for_one_round 

        results= {}
        for t in thresholds:
            results_at_t= []
            for i in meta_metrics.keys():
                detected = mal_agents_val_accuracy(meta_metrics[i], t)      #an array with 1s at mal agent indices
                results_at_t.append(detected)
            results[str(np.round(t,3))] = results_at_t

        with open(f'detection_results/val_accu_detection_results_r{rvalue}.pkl', 'wb') as handle:
            pickle.dump(results,handle,protocol=pickle.HIGHEST_PROTOCOL)
        print("written for one rvalue")


def draw_f1_vs_threshold (dict_l_f1):
    print('plotting')
    plt.figure(figsize=(8,6))
    
    #ax1 = fig1.add_subplot(1,1,1)
    
    for rvalue, list_f1 in dict_l_f1.items():
        plt.plot(thresholds, list_f1, label= f'R={str(int(rvalue))}')
    
    plt.xlabel('Thresholds', fontsize=14)
    plt.ylabel('F1 score', fontsize=14)
    plt.legend()
    plt.savefig('plots/f1_vs_thresholds.png')

def draw_roc_curve(dict_l_tpr, dict_l_fpr, dict_l_precision):

    fig1 = plt.figure(figsize=(10,5))
    ax1 = fig1.add_subplot(1,2,1)
    ax2 = fig1.add_subplot(1,2,2)

    add_text= 1
    for (rvalue, l_TPR), (rvalue, l_FPR) in zip(dict_l_tpr.items(), dict_l_fpr.items()):
        ax1.plot(l_FPR, l_TPR, label=f'R={str(int(rvalue))}')
        ax1.plot([0,1], [0,1], color = 'r', ls='--')
        if(add_text == 1):
            add_text= 0
            for i, (FPR, TPR) in enumerate(zip(l_FPR, l_TPR)):
                if (i % 2 == 0 or i == 5) and (i != 10):
                    ax1.text(FPR, TPR, thresholds[i], horizontalalignment='center', verticalalignment='bottom')
                if i > 15:
                    break
    ax1.set_xlabel("False Positive Rate", fontsize = 14)
    ax1.set_ylabel("True Positive Rate", fontsize = 14)
    
    #ax2
    add_text=1
    for (rvalue, l_TPR), (rvalue, l_prec) in zip(dict_l_tpr.items(), dict_l_precision.items()):
        ax2.plot(l_TPR, l_prec)
        ax2.plot([1,0], [0,1], color = 'r', ls='--')

        if(add_text == 1):
            add_text= 0
            for i, (TPR, prec) in enumerate(zip(l_TPR, l_prec)):
                if (i % 2 == 0 or i == 5) and (i != 10):
                    if(TPR > 0.95):
                        ax2.text(TPR+.04, prec, thresholds[i], horizontalalignment='center', verticalalignment='bottom', fontsize='xx-small')
                    else:
                        ax2.text(TPR, prec, thresholds[i], horizontalalignment='center', verticalalignment='bottom')
                if i > 15:
                    break
    ax2.yaxis.set_label_position("right")
    ax2.set_xlabel("Recall", fontsize = 14)
    ax2.set_ylabel("Precision", fontsize = 14)

    fig1.legend(loc=4)
    fig1.savefig('plots/roc.png')

def wrapper_for_plots():
    #now load the results and plot necessary graphs
    list_of_results_dicts= []       #list_of_results_dicts[0]= a dict that contains the results for diff thresholds for rvalues[0]
    Y_true= [0]*NUMBER_OF_CLIENTS
    for m in MAL_CLIENT_INDICES:
        Y_true[m]= 1
    Y_true_mal= Y_true * NUM_OF_ROUNDS

    for rvalue in rvalues:    
        with open(f'detection_results/val_accu_detection_results_r{rvalue}.pkl', 'rb') as handle:
            results_dict= pickle.load(handle)
        #print(results_dict.keys())
        list_of_results_dicts.append(results_dict)
    #print(list_of_results_dicts[0]['0.025'])


    rvalue_index= 0
    dict_of_l_accu= {}
    dict_of_l_TPR= {}
    dict_of_l_FPR= {}
    dict_of_l_f1= {}
    dict_of_l_precision= {}

    for results_dict in list_of_results_dicts:
        #if (rvalues[rvalue_index]=='00000'): Y_true= [0]*NUMBER_OF_CLIENTS*NUM_OF_ROUNDS
        #else: Y_true= Y_true_mal.copy()
        if (rvalues[rvalue_index]=='00000'): 
            rvalue_index += 1
            continue           #tpr gets divide be zero
        Y_true= Y_true_mal.copy()
        #combine results
        results_uni = {}
        for key in results_dict.keys():
            results_uni[key] = np.concatenate(results_dict[key])

        l_accu = []
        l_TPR = []
        l_FPR = []
        l_f1 = []
        l_precision= []

        for key in results_uni.keys():
            accu = accuracy_score(Y_true, results_uni[key])
            cmtx = confusion_matrix(Y_true, results_uni[key], labels=[True, False])
            TN = cmtx[0,0]
            FN = cmtx[1,0]
            TP = cmtx[1,1]
            FP = cmtx[0,1]
            TPR = TP / (TP+FN)      #recall
            FPR = FP / (FP+TN)
            precision= TP/ (TP+FP)
            F1= 2*TP/(2*TP + FP + FN)

            l_accu.append(accu)
            l_TPR.append(TPR)
            l_FPR.append(FPR) 
            l_f1.append(F1)
            l_precision.append(precision)

        dict_of_l_accu[rvalues[rvalue_index]]= l_accu 
        dict_of_l_TPR[rvalues[rvalue_index]]= l_TPR
        dict_of_l_FPR[rvalues[rvalue_index]]= l_FPR 
        dict_of_l_f1[rvalues[rvalue_index]]= l_f1
        dict_of_l_precision[rvalues[rvalue_index]]= l_precision
        rvalue_index += 1

    return dict_of_l_f1, dict_of_l_TPR, dict_of_l_FPR, dict_of_l_precision

def plot_accu_vs_rounds():
    dict_of_list_of_accus_for_one_rvalue= {}
    
    for rvalue in rvalues:
        list_of_accus= []
        hist_filename= '/home/salesort/Documents/644_CMPUT/outputs/Results_r' + rvalue +'/hist.pkl'
        with open(hist_filename, 'rb') as handle:
            hist_for_one_r = pickle.load(handle)
        for acc in hist_for_one_r.metrics_distributed['accuracy']:
            list_of_accus.append(acc[1])           #20 (#rounds) accu value for one rvalue
        
        dict_of_list_of_accus_for_one_rvalue[rvalue]= list_of_accus
        
    #now plot
    plt.figure(figsize=(8,6))
    x= list(range(1,21)) #np.arange(1,21,1, dtype=int)
    for r, l_accus in dict_of_list_of_accus_for_one_rvalue.items():
        if(int(r)==0 or int(r)==1 or int(r)==100 or int(r)==1000 or int(r)==3000):
            plt.plot(x, l_accus, label=f'R={str(int(r))}')
    plt.legend()
    plt.grid()
    plt.xticks(np.arange(min(x), max(x)+1, 4))
    plt.xlabel('Rounds', fontsize=14)
    plt.ylabel('Model accuracy', fontsize=14)
    plt.savefig('plots/model_accu_vs_rounds_different_r.png')

        
    
#save_results_mal_agents_with_saved_parameters()        #run this once to save the detection results  
dict_of_l_f1, dict_of_l_TPR, dict_of_l_FPR, dict_of_l_precision= wrapper_for_plots()

##draw thershold vs F1 score plot
draw_f1_vs_threshold(dict_of_l_f1)

##draw roc
#draw_roc_curve(dict_of_l_TPR, dict_of_l_FPR, dict_of_l_precision)

#draw accu vs rounds curve
#plot_accu_vs_rounds()






    
