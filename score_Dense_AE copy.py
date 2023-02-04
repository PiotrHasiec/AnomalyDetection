import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import  MinMaxScaler
from scipy.signal import savgol_filter as savgol_filter
from ModelLoader import ModelLoader
from DataLoader import DataLoader
from Evaluator import Evaluator
from functions import n
EPOCHES = 50
TIME_STEP = 100
JOINTS = ['sLAnkleAngles','cLAnkleAngles','pLAnkleAngles','sLHipAngles','cLHipAngles','pLHipAngles','sLKneeAngles','sLWristAngles','cLWristAngles','pLWristAngles','sRAnkleAngles','cRAnkleAngles','pRAnkleAngles','sRHipAngles','cRHipAngles','pRHipAngles','sRKneeAngles','sRWristAngles','cRWristAngles','pRWristAngles']





if __name__ == '__main__':
    path = ".\\Modele\\Denoisy_AE4_B04all"

    for actor in range(6,20):
        print(actor)
        
        test_loader = DataLoader()
        model_loader = ModelLoader()
        anomally_loader = DataLoader(test_size=0)
        
        test_files = os.listdir("Dataset\\Caren\\Connected\\normalized\\Train\\")
        test_datas = [pd.read_csv("Dataset\\Caren\\Connected\\normalized\\Train\\"+file_name).get(JOINTS) for file_name in test_files if file_name.startswith('B04'+n(actor))]
        test_datas = [data for data in test_datas if not data is None]
        if len(test_datas) == 0:
            continue
        train_x,test_x,train_y,test_y = test_loader.load_data(test_datas,1,100,100)

        anomally_files = os.listdir("Dataset\\Caren\\Connected\\normalized\\Annomaly\\")
        # for file in anomally_files:
        # if not file.startswith('B04'+n(actor)):
        #     continue
        # anomally_datas = [pd.read_csv("Dataset\\Caren\\Connected\\normalized\\Annomaly\\"+file)]
        anomally_datas = [pd.read_csv("Dataset\\Caren\\Connected\\normalized\\Annomaly\\"+file_name,index_col=False) for file_name in anomally_files if file_name.startswith('B04'+n(actor))]#normalized\\Train\\
        anomally_datas = [data for data in anomally_datas if not data is None]
        anomally_x,_,_,_ = anomally_loader.load_data([a.get(JOINTS) for a in anomally_datas],1,100,100)
        eval = Evaluator((test_x,test_x),(anomally_x,anomally_x))
        model = model_loader.Load_model('AE_model',path)
        mean,std,treshold_min,treshold_max = eval.find_treshold(model)
        tresholds = np.linspace(treshold_min-std,treshold_max+std,10)
        pr4all = []
        pr4visible =[]
        f1s4all = []
        f1s4visible = []
        y_true = pd.DataFrame()
        for data in anomally_datas:
            y_true = pd.concat([y_true,data.get(['Type','Visible'])[:(data.shape[0]//100)*100] ],ignore_index=True)
        try:

            for treshold in tresholds:
                f14all,prec4all,rec4all,f14visible,prec4cisible,rec4visible = eval.score(model,treshold_max,y_true)
                pr4all.append([rec4all,prec4all])
                pr4visible.append([rec4visible,prec4cisible])
                f1s4all.append(f14all)
                f1s4visible.append(f14visible)
                print(treshold)
                eval.visualise(model,treshold)  
            # eval.visualise(model,tresholds[np.argmax(f1s4all)])    
            pr = np.transpose(np.array(pr4all))

            plt.plot(pr[0],pr[1])
            pr = np.transpose(np.array(pr4visible))
        
            plt.plot(pr[0],pr[1])
            plt.show()

        except Exception as inst:
            print(inst)
            continue

