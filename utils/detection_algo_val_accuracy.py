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
domPropValues= ['10', '50', '75', '90', '99']
thresholds= np.round(np.concatenate((np.arange(.01, .05, 0.01), np.arange(.08, .15, 0.02))),3)
#print(thresholds)

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
    #saved matrices (actually dictionaries) to check detection strategies performance and generate 'detected' arrays
    
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

        with open(f'detection_results_trial2/val_accu_detection_results_r{rvalue}.pkl', 'wb') as handle:
            pickle.dump(results,handle,protocol=pickle.HIGHEST_PROTOCOL)
        print("written for one rvalue")

def save_detection_results_for_unbalanced_data():
    #use saved parameters for the detection algorithm. Save the detection results for plotting afterwards.

    for rvalue in ['00001']:        #we've done it for one ravlue for now.
        for domprop in domPropValues:    
            param_path= '/home/salesort/Documents/644_CMPUT/outputs/Results_r' + rvalue + '_prop0-'+ domprop + '/'
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

            with open(f'detection_results_trial2/val_accu_detection_results_r{rvalue}_prop0_{domprop}.pkl', 'wb') as handle:
                pickle.dump(results,handle,protocol=pickle.HIGHEST_PROTOCOL)
            print("written for one domprop value")
        print("written for one rvalue")

def draw_f1_vs_threshold (dict_l_f1, unbalanced=0):
    print('plotting')
    plt.figure(figsize=(8,6))
    
    #ax1 = fig1.add_subplot(1,1,1)
    
    for rvalue, list_f1 in dict_l_f1.items():
        if (unbalanced == 0): plt.plot(thresholds, list_f1, label= f'R={str(int(rvalue))}')
        else: plt.plot(thresholds, list_f1, label= f'DomProp={str(int(rvalue))}')
    
    plt.xlabel('Thresholds', fontsize=14)
    plt.ylabel('F1 score', fontsize=14)
    plt.legend()
    if(unbalanced==0): plt.savefig('plots/f1_vs_thresholds.png')
    else: plt.savefig('plots/f1_vs_thresholds_unbalanced.png')

def draw_roc_curve(dict_l_tpr, dict_l_fpr, dict_l_precision, unbalanced=0):

    fig1 = plt.figure(figsize=(10,5))
    ax1 = fig1.add_subplot(1,2,1)
    ax2 = fig1.add_subplot(1,2,2)
    for (rvalue, l_TPR), (rvalue, l_FPR) in zip(dict_l_tpr.items(), dict_l_fpr.items()):
        if(unbalanced == 0): ax1.plot(l_FPR, l_TPR, label=f'R={str(int(rvalue))}')
        else: ax1.plot(l_FPR, l_TPR, label=f'DomProp={str(float(rvalue)/100)}')
        ax1.plot([0,1], [0,1], color = 'r', ls='--')
        if(rvalue=='10'):      #write ALL texts in BLUE
            for i, (FPR, TPR) in enumerate(zip(l_FPR, l_TPR)):
                ax1.text(FPR-.01, TPR+.005, thresholds[i], horizontalalignment='center', verticalalignment='bottom', color='blue', fontsize='xx-small', rotation=90)
                if (FPR >= 1):
                    break
        elif (rvalue== '50'):   #write ALL texts in ORANGE
            for i, (FPR, TPR) in enumerate(zip(l_FPR, l_TPR)):
                    if( thresholds[i] < .08): ax1.text(FPR-.03, TPR, thresholds[i], horizontalalignment='center', verticalalignment='bottom', color='darkorange', fontsize='x-small', rotation=45)
                    elif(thresholds[i] == .08): ax1.text(FPR+.01, TPR, thresholds[i], horizontalalignment='center', verticalalignment='bottom', color='darkorange', fontsize='x-small', rotation=45)
                    elif (thresholds[i] > .08): ax1.text(FPR, TPR-.04, thresholds[i], horizontalalignment='center', verticalalignment='bottom', color='darkorange', fontsize='x-small')
                    if (FPR >= 1):
                        break
        elif (rvalue== '75'):   #write SOME texts in GREEN
            for i, (FPR, TPR) in enumerate(zip(l_FPR, l_TPR)):
                    if(thresholds[i] < .08): continue 
                    elif(thresholds[i] == .08): ax1.text(FPR+.04, TPR, thresholds[i], horizontalalignment='center', verticalalignment='bottom', color='green', fontsize='x-small', rotation=45)
                    else: ax1.text(FPR+.04, TPR, thresholds[i], horizontalalignment='center', verticalalignment='bottom', color='green', fontsize='x-small')
                    #elif (thresholds[i] > .08): ax1.text(FPR, TPR-.04, thresholds[i], horizontalalignment='center', verticalalignment='bottom', color='darkorange', fontsize='x-small')
                    if (FPR >= 1):
                        break
        elif(rvalue == '99'):
            for i, (FPR, TPR) in enumerate(zip(l_FPR, l_TPR)):
                    if(thresholds[i] < .14): continue
                    else: 
                        ax1.text(FPR+.04, TPR, thresholds[i], horizontalalignment='center', verticalalignment='bottom', color='darkviolet', fontsize='x-small')


    ax1.set_xlabel("False Positive Rate", fontsize = 14)
    ax1.set_ylabel("True Positive Rate", fontsize = 14)
    
    #ax2
    for (rvalue, l_TPR), (rvalue, l_prec) in zip(dict_l_tpr.items(), dict_l_precision.items()):
        ax2.plot(l_TPR, l_prec)
        ax2.plot([1,0], [0,1], color = 'r', ls='--')

        if(rvalue == '10'):
            for i, (TPR, prec) in enumerate(zip(l_TPR, l_prec)):
                if(thresholds[i] <=.02):
                    ax2.text(TPR+.05, prec, thresholds[i], horizontalalignment='center', verticalalignment='bottom', fontsize='x-small', rotation=40, color='darkblue')
                elif(thresholds[i] >= .14):
                    ax2.text(TPR+.03, prec, thresholds[i], horizontalalignment='center', verticalalignment='bottom', fontsize='x-small', color='darkblue', rotation=40)
        elif(rvalue == '50'):
            for i, (TPR, prec) in enumerate(zip(l_TPR, l_prec)):
                if(thresholds[i] <=.04):
                    ax2.text(TPR, prec, thresholds[i], horizontalalignment='center', verticalalignment='bottom', fontsize='x-small', color='darkorange', rotation=45)
                elif(thresholds[i] ==0.14):
                    ax2.text(TPR, prec-.04, thresholds[i], horizontalalignment='center', verticalalignment='bottom', fontsize='x-small', color='darkorange')
        
        elif(rvalue == '75'):
            for i, (TPR, prec) in enumerate(zip(l_TPR, l_prec)):
                if(thresholds[i] ==.08):
                    ax2.text(TPR+.03, prec-.03, thresholds[i], horizontalalignment='center', verticalalignment='bottom', fontsize='x-small', color='green')
                elif(thresholds[i]  ==.1):
                    ax2.text(TPR+.03, prec-.04, thresholds[i], horizontalalignment='center', verticalalignment='bottom', fontsize='x-small', color='green')
                elif(thresholds[i]  ==.14):
                    ax2.text(TPR+.03, prec-.03, thresholds[i], horizontalalignment='center', verticalalignment='bottom', fontsize='x-small', color='green')
                
    ax2.yaxis.set_label_position("right")
    ax2.set_xlabel("Recall", fontsize = 14)
    ax2.set_ylabel("Precision", fontsize = 14)

    fig1.legend(loc=4)
    if(unbalanced==0): fig1.savefig('plots/roc.png')
    else: fig1.savefig('plots/roc_unbalanced.png')

def wrapper_for_plots(unbalanced=0):
    #now load the results and plot necessary graphs
    list_of_results_dicts= []       #list_of_results_dicts[0]= a dict that contains the results for diff thresholds for rvalues[0]
    Y_true= [0]*NUMBER_OF_CLIENTS
    for m in MAL_CLIENT_INDICES:
        Y_true[m]= 1
    Y_true_mal= Y_true * NUM_OF_ROUNDS

    if(unbalanced == 0):
        for rvalue in rvalues:    
            with open(f'detection_results_trial2/val_accu_detection_results_r{rvalue}.pkl', 'rb') as handle:
                results_dict= pickle.load(handle)
            #print(results_dict.keys())
            list_of_results_dicts.append(results_dict)
        #print(list_of_results_dicts[0]['0.025'])
    else:
        for rvalue in ['00001']:    
            for domprop in domPropValues:    
                with open(f'detection_results_trial2/val_accu_detection_results_r{rvalue}_prop0_{domprop}.pkl', 'rb') as handle:
                    results_dict= pickle.load(handle)
                #print(results_dict.keys())
                list_of_results_dicts.append(results_dict)


    rvalue_index= 0                     #tpr get divide by 0 if we start from rvalue_index=0 
    domprop_index= 0
    dict_of_l_accu= {}
    dict_of_l_TPR= {}
    dict_of_l_FPR= {}
    dict_of_l_f1= {}
    dict_of_l_precision= {}

    for results_dict in list_of_results_dicts:
        if(unbalanced ==0 and rvalue_index== 0):
            rvalue_index+= 1
            continue
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

        if(unbalanced == 0): 
            key= rvalues[rvalue_index]
            rvalue_index += 1
        else: 
            key= domPropValues[domprop_index]
            domprop_index += 1
        
        dict_of_l_accu[key]= l_accu 
        dict_of_l_TPR[key]= l_TPR
        dict_of_l_FPR[key]= l_FPR 
        dict_of_l_f1[key]= l_f1
        dict_of_l_precision[key]= l_precision

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

        
##BALANCED
#save_results_mal_agents_with_saved_parameters()        #run this once to save the detection results  
##UNBALANCED
#save_detection_results_for_unbalanced_data()            #run this once to save the detection results 

##draw accu vs rounds curve
#plot_accu_vs_rounds()



dict_of_l_f1, dict_of_l_TPR, dict_of_l_FPR, dict_of_l_precision= wrapper_for_plots(1)
draw_f1_vs_threshold(dict_of_l_f1,1)
draw_roc_curve(dict_of_l_TPR, dict_of_l_FPR, dict_of_l_precision,1)




    
