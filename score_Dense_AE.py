import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import  MinMaxScaler
from scipy.signal import savgol_filter as savgol_filter
from ModelLoader import ModelLoader
from DataLoader import DataLoader
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,average_precision_score,precision_recall_curve,roc_curve
from functions import *
from scipy.integrate import trapezoid
EPOCHES = 100
TIME_STEP = 50
DETECTING_STEP = 50
JOINTS = ['sLAnkleAngles','cLAnkleAngles','pLAnkleAngles','sLHipAngles','cLHipAngles','pLHipAngles','sLKneeAngles','sLWristAngles','cLWristAngles','pLWristAngles','sRAnkleAngles','cRAnkleAngles','pRAnkleAngles','sRHipAngles','cRHipAngles','pRHipAngles','sRKneeAngles','sRWristAngles','cRWristAngles','pRWristAngles']

def find_treshold(model,path,condition):
    joints = ['sLAnkleAngles','cLAnkleAngles','pLAnkleAngles','sLHipAngles','cLHipAngles','pLHipAngles','sLKneeAngles','sLWristAngles','cLWristAngles','pLWristAngles','sRAnkleAngles','cRAnkleAngles','pRAnkleAngles','sRHipAngles','cRHipAngles','pRHipAngles','sRKneeAngles','sRWristAngles','cRWristAngles','pRWristAngles']
    model.build((TIME_STEP,len(joints)))
    model.compile(loss =tf.losses.mean_squared_error,optimizer= tf.keras.optimizers.Adadelta(learning_rate=1))



    files = os.listdir("Dataset\\Caren\\Connected\\normalized\\Train\\")
   



    test_datas = [pd.read_csv("Dataset\\Caren\\Connected\\normalized\\Train\\"+file_name).get(joints) for file_name in files if condition(file_name)]
    test_datas = [data for data in test_datas if not data is None]
    test_loader = DataLoader()

    if len(test_datas) == 0:
            return
    train_x,test_x,train_y,test_y = test_loader.load_data(test_datas,1,TIME_STEP,TIME_STEP)
    timestep_mean_errors = []

    try:           
        predicted_data = model.predict(test_x,verbose=0)
    except Exception as inst:
                    print(type(inst))    # the exception instance
                    print(inst.args)     # arguments stored in .args
                    print(inst) 
    predicted_data = predicted_data.reshape((-1,len(joints)))
    all_errors = (test_x.reshape((-1,len(joints))) - predicted_data)**2
    errors = savgol_filter(np.amax(all_errors,1),60,3)
    # timestep_mean_errors+=errors
            
           


    counts, bins = np.histogram(errors,30)

    plt.stairs(counts, bins)
    # plt.show()
    plt.savefig(path)
    plt.clf()
    mean = np.mean(errors)
    std = np.std(errors)
    return (mean,std,np.percentile(errors,0),np.percentile(errors,100))

def p_r_f1(y_true,y_pred):
        gruped_true = [y_true[0]]
        gruped_pred = [y_pred[0]]
        try:
            for i in range(DETECTING_STEP,len(y_true)-1):
                if y_pred[i]!=gruped_pred[-1] or y_true[i]!=gruped_true[-1]:
                    # if gruped_pred[-1] == y_true[-1] and gruped_pred[-1] ==True:
                    #     continue
                    # else:
                    # if i%DETECTING_STEP ==0:
                        gruped_true.extend([np.max(y_true[i-DETECTING_STEP:i])])
                        gruped_pred.extend([np.max(y_pred[i-DETECTING_STEP:i])])
        except Exception as i:
            print(i.args)
            print(i)
        return f1_score(y_true,y_pred,zero_division=1),precision_score(y_true,y_pred,zero_division=1),recall_score(y_true,y_pred,zero_division=0)




def score(model,condition):
    joints = ['sLAnkleAngles','cLAnkleAngles','pLAnkleAngles','sLHipAngles','cLHipAngles','pLHipAngles','sLKneeAngles','sLWristAngles','cLWristAngles','pLWristAngles','sRAnkleAngles','cRAnkleAngles','pRAnkleAngles','sRHipAngles','cRHipAngles','pRHipAngles','sRKneeAngles','sRWristAngles','cRWristAngles','pRWristAngles']
    y = []
    y_d = []
    x_d = []
    x = []
    y_true = []
    errors =np.array([])
    for filename in os.listdir("Dataset\\Caren\\Connected\\normalized\\Annomaly\\") :
        if filename.endswith('.csv') and condition(filename):
            try:   


                    data = pd.read_csv("Dataset\\Caren\\Connected\\normalized\\Annomaly\\"+filename ).get(joints)
                    data = data.get(joints)
                    s = model.input_shape[1]
                    x = np.array(data)[:(data.shape[0]//s)*s]
                    x =x.reshape(-1,s,len(joints))
                    
                    predicted_data = model.predict(x,verbose=0).reshape((-1,len(joints)))
                    all_errors = (np.abs((x.reshape(-1,len(joints)) - predicted_data)))
                    current_errors = np.mean(all_errors,1)
                    errors = np.append(errors, current_errors )
                    

                    try:
                        z = pd.read_csv("Dataset\\Caren\\Connected\\normalized\\Annomaly\\"+filename ).get(["Type","Visible"])
                        z = z.fillna("")

                        dic = {"A":0.1,"B":0.2,"C":0.3,'':0,'B1':0.2,'B2':0.2,'B3':0.2,'O':0.4,}
                        for j,err in enumerate(current_errors):
                            type_of_disturbance = z['Type'][j]
                            

                            if type_of_disturbance !='':
                                if z['Visible'][j] or True :#or err > treshold:
                                    y_true.append(True)
                                else:
                                    y_true.append(False)
                            else:
                                y_true.append(False)
                            
                    except Exception as inst:
                        # print(inst)
                        print("")
                    plt.xlim((0,len(errors)))
                    plt.ylim(bottom=0)
                        

                    
                    
                    e_visible = [ (i,dic[typ]) for i,typ in enumerate(z['Type']) if z['Visible'][i]]
                    e_all = [ (i,dic[typ]) for i,typ in enumerate(z['Type'])]

            except Exception as inst:
                    continue

    y_tr = []
    for y in y_true:
        if y:
            y_tr.append(1)
        else:
            y_tr.append(0)
    gruped_true = [y_tr[0]]
    gruped_pred = [errors[0]]
    
    for i in range(DETECTING_STEP,len(y_tr)-DETECTING_STEP):
        # if y_pred[i]!=gruped_pred[-1] or y_true[i]!=gruped_true[-1]:
            # if gruped_pred[-1] == y_true[-1] and gruped_pred[-1] ==True:
            #     continue
            # else:
            # if i%DETECTING_STEP ==0:
                gruped_true.extend([np.max(y_tr[i-DETECTING_STEP:i+DETECTING_STEP])])
                gruped_pred.extend([np.max(errors[i-DETECTING_STEP:i+DETECTING_STEP])])         


    return gruped_true,gruped_pred




def score_in_treshold_function(path,actor,model):
    pr = []
    f1s = []
    condition = lambda x: x.startswith('B04'+actor)
    mean,std,treshold_min,treshold_max = find_treshold(model,path+"\\histogram.png",condition)
    tresholds= np.linspace(treshold_min,treshold_max+6*std,20)
    # tresholds=[mean+3*std]
    statistic = np.array([0,0,0])
    for treshold in tresholds:
        f1,prec,recall, stats = score(model,treshold,condition)
        # statistic+=stats
        pr.append([prec,recall])
        f1s.append([f1,treshold])
        print(pr)
        print(f1s)
        print(treshold)
    
    pr.sort(key=lambda x: x[1])
    tosave = pd.DataFrame(pr)
    tosave.to_csv(path+"\\PR_"+actor+".csv",sep = ';',decimal = ',')
    precisions = [i[0] for i in pr ]
    # reinterpreted_precision = [np.max(precisions[i:])for i in range(len(precisions))]
    recalls = [i[1] for i in pr]
    precisions =  [1]+ precisions + [precisions[-1]]
    reinterpreted_precision = [np.max(precisions[i:])for i in range(len(precisions))]
    recalls =  [recalls[0]]+ recalls + [1]
    Ap = trapezoid(precisions,recalls)/(np.max(recalls)- np.min(recalls))
    RAp = trapezoid(reinterpreted_precision,recalls)/(np.max(recalls)- np.min(recalls))
    print('ap: {}'.format(Ap))
    print('rap: {}'.format(RAp))
    # plt.clf()
    # plt.plot(recalls,precisions)
    # plt.show()

    plt.clf()
    f1s.sort(key=lambda x: x[1])
    tosave = pd.DataFrame(f1s)
    tosave.to_csv(path+"\\F1.csv",sep = ';',decimal = ',')
    f1 = [i[0] for i in f1s ]
    tr = [i[1] for i in f1s]
    # plt.plot(tr ,f1)
    # plt.show()
    return Ap,RAp
TIME_STEP = 100
# if __name__ == 'score':".\\Modele\\Denoisy_AE50_B04all",
paths = [".\\Modele\\Denoisy_AE50_B04all",".\\Modele\\Denoisy_AE25_B04all",]
in_latendim = []
for laten_dim in [30]:
    in_noise =[]
    for noise in [0.05]:#,0.1,0.3,0.5,0.7]:
        gp,gt =[],[]
        aucs = []
        pr = []
        actors =[]
        w = []
        path = '.\\Modele\AE_ldim_{}_noise_{}_timsetps_{}'.format(laten_dim,noise,TIME_STEP)
        try:
            model_loader = ModelLoader()
            model = model_loader.Load_model('AE_model',path)
        except:
            print("Nie udało się załadować modelu")
            continue
        for actor in range(6,20):
            # print(actor)
            if actor in [8,15,19]:
                continue
            # anomally_loader = DataLoader(test_size=0)

            # files = os.listdir("Dataset\\Caren\\Connected\\normalized\\Train\\")#

            # datas = [pd.read_csv("Dataset\\Caren\\Connected\\normalized\\Train\\"+file_name).get(JOINTS) for file_name in files if file_name.startswith('B04'+n(actor))]#normalized\\Train\\
            # datas = [data for data in datas if not data is None]
            try:
                
                condition = lambda x: x.startswith('B04'+n(actor))
                g_t,g_p = score(model,condition)
                gt.extend(g_t)
                gp.extend(g_p)
                # actors.append(actor)
            except Exception as inst:
                print(inst)
                continue
        # print(aps)
        # print(aucs)
        print(average_precision_score(y_true=gt,y_score=gp))
        prec,rec,tresh = precision_recall_curve(y_true=gt,probas_pred = gp)
        plt.cla()
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.plot(rec,prec)
        plt.show()
        Tpr,Fpr,tresh = roc_curve(y_true=gt,y_score=gp)
        plt.cla()
        plt.xlabel("TPR")
        plt.ylabel("FPR")
        plt.plot(Tpr,Fpr)
        plt.plot([0,1],[0,1])
        plt.show()


