import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from Model import build_model
from sklearn.preprocessing import  MinMaxScaler
from scipy.signal import savgol_filter as savgol_filter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  MinMaxScaler
from keras.utils import timeseries_dataset_from_array
from Train_model_SingleStepPrediction import load_data,TIME_STEP
import Model
EPOCHES = 50
NUMBER_OF_TIMESTEPS_IN_SERIE = 50

def n(number):
    if number<10:
        return '0'+str(number)
    return str(number)

def Load_model(path = ".\\Modele\\M1"):
    
    name_weights = ''
    csv_data = None
    for file in os.listdir(path):
        if os.path.isfile(path+'\\'+file) and file.endswith('.csv') and file.startswith('prediction_model'):
            csv_data = pd.read_csv(path+'\\'+file)
        if os.path.isfile(path+'\\'+file) and file.endswith('.json'):
            with open(path+'\\'+file, 'r') as plik:
                loaded_model_json = plik.read()
                model = tf.keras.models.model_from_json(loaded_model_json)
        if os.path.isfile(path+'\\'+file) and file.endswith('.h5'):
            try:
                if file.split('_')[-1].split('.')[0] > name_weights.split('_')[-1].split('.')[0]:
                    name_weights = file
            except:
                continue
    if not csv_data is None:
            name_parts = name_weights.split('.h5')[0].split('_')[:-1]
            name =''
            for part in name_parts:
                name += part+'_'
            name  += str(csv_data[csv_data.columns[-1]].argmin())
            model.load_weights(path+'\\'+name+'.h5')
    joints = ['sLAnkleAngles','cLAnkleAngles','pLAnkleAngles','sLHipAngles','cLHipAngles','pLHipAngles','sLKneeAngles','sLWristAngles','cLWristAngles','pLWristAngles','sRAnkleAngles','cRAnkleAngles','pRAnkleAngles','sRHipAngles','cRHipAngles','pRHipAngles','sRKneeAngles','sRWristAngles','cRWristAngles','pRWristAngles']
    model.build((100,NUMBER_OF_TIMESTEPS_IN_SERIE,len(joints)))
    model.compile(loss =tf.losses.mean_squared_error,optimizer= tf.keras.optimizers.Adadelta(learning_rate=1))

    return model

    

def find_treshold(model,path,condition):
    joints = ['sLAnkleAngles','cLAnkleAngles','pLAnkleAngles','sLHipAngles','cLHipAngles','pLHipAngles','sLKneeAngles','sLWristAngles','cLWristAngles','pLWristAngles','sRAnkleAngles','cRAnkleAngles','pRAnkleAngles','sRHipAngles','cRHipAngles','pRHipAngles','sRKneeAngles','sRWristAngles','cRWristAngles','pRWristAngles']
    model.build((100,NUMBER_OF_TIMESTEPS_IN_SERIE,len(joints)))
    model.compile(loss =tf.losses.mean_squared_error,optimizer= tf.keras.optimizers.Adadelta(learning_rate=1))



    files = os.listdir("Dataset\\Caren\\Connected\\normalized\\Train\\")
   



    datas = [pd.read_csv("Dataset\\Caren\\Connected\\normalized\\Train\\"+file_name).get(joints) for file_name in files if condition(file_name)]
    datas = [data for data in datas if not data is None]
    train_x,test_x,train_y,test_y = load_data(datas)
    timestep_mean_errors = []

                
    predicted_data = model.predict(test_x,verbose=0).reshape((-1,len(joints)))
    all_errors = (test_y.reshape((-1,len(joints))) - predicted_data)**2
    errors = [np.sqrt(np.sum(err)) for err in all_errors]
    timestep_mean_errors+=errors
            
           


    counts, bins = np.histogram(timestep_mean_errors,30)

    plt.stairs(counts, bins)
    # plt.show()
    plt.savefig(path)
    plt.clf()
    mean = np.mean(timestep_mean_errors)
    std = np.std(timestep_mean_errors)
    return (mean,std,np.percentile(timestep_mean_errors,0),np.percentile(timestep_mean_errors,100))

def score(model, treshold,condition):
    tp = [0,0]
    fp = [0,0]
    fn = [0,0]
    joints = ['sLAnkleAngles','cLAnkleAngles','pLAnkleAngles','sLHipAngles','cLHipAngles','pLHipAngles','sLKneeAngles','sLWristAngles','cLWristAngles','pLWristAngles','sRAnkleAngles','cRAnkleAngles','pRAnkleAngles','sRHipAngles','cRHipAngles','pRHipAngles','sRKneeAngles','sRWristAngles','cRWristAngles','pRWristAngles']
    for filename in os.listdir("Dataset\\Caren\\Connected\\normalized\\Annomaly\\") :
        if filename.endswith('.csv') and condition(filename):
            try:   

                    print(filename)
                    data = pd.read_csv("Dataset\\Caren\\Connected\\normalized\\Annomaly\\"+filename ).get(joints)
                    data = data.get(joints)
                    x = np.array(data[:-TIME_STEP]).reshape(1,-1,len(joints))
                    y = np.array(data[TIME_STEP:]).reshape(-1,len(joints))
                    predicted_data = model.predict(x,verbose=0).reshape((-1,len(joints)))
                    all_errors = (y  - predicted_data)**2
                    errors = [np.sqrt(np.sum(err)) for err in all_errors]
                    
                    y = []
                    y_d = []
                    x_d = []
                    x = []
                    z = pd.read_csv("Dataset\\Caren\\Connected\\normalized\\Annomaly\\"+filename ).get(["Type","Visible"])
                    z = z.fillna("")
                    detected:bool = False
                    disturbance:bool = False
                    visible:bool = False

                    dic = {"A":0.1,"B":0.2,"C":0.3,'':0,'B1':0.2,'B2':0.2,'B3':0.2,'O':0.4,}
                    
                    


                    for j,err in enumerate(errors):
                        type_of_disturbance = z['Type'][j]
                        

                        if type_of_disturbance !='':
                            disturbance = True
                            visible = visible or z['Visible'][j]
                        y.append((err) )
                        x.append(j)
                        if  (err) > treshold:#mean+2.57*std:
                        
                                y_d.append(err)
                                x_d.append(j)
                                detected = True
                        else:
                            if not disturbance and detected:
                                    fp[0]+=1
                                    detected = False
                            
                        if type_of_disturbance =='' and disturbance == True:
                            if detected:
                                if visible==True:
                                                tp[0]+=1
                                else:
                                                tp[0]+=1
                                                tp[1]+=1
                            else:
                                if visible==True:
                                                fn[0]+=1
                                else:
                                                fn[1]+=1
                            visible = False
                            disturbance = False
                            detected = False

                                    

                    plt.scatter(x,y,color = 'g')
                    plt.scatter(x_d,y_d,color = 'r')
                    plt.xlim((0,len(errors)))
                    plt.ylim(bottom=0)

                    dic = {"A":0.1,"B":0.2,"C":0.3,'':0,'B1':0.2,'B2':0.2,'B3':0.2,'O':0.4,}
                    
                    
                    e_visible = [ (i,dic[typ]) for i,typ in enumerate(z['Type']) if z['Visible'][i]]
                    e_all = [ (i,dic[typ]) for i,typ in enumerate(z['Type'])]

                    x_all,e_all = list(zip(*e_all))
                    plt.scatter(x_all,e_all,color="m")
                    if len(e_visible)>0:
                        x_vis,e_visible = list(zip(*e_visible))
                        plt.scatter(x_vis,e_visible,color="b")
                    # plt.show()
            except Exception as inst:
                    print(type(inst))    # the exception instance
                    print(inst.args)     # arguments stored in .args
                    print(inst) 
                
                    
                    
                    # plt.show()
    try:
                        print('f1:',(2*tp[0])/(2*tp[0]+fp[0]+fn[0]))
                        print('f1:',(2*tp[1])/(2*tp[1]+fp[0]+fn[1]))

                        print('precision for visible',tp[0]/(tp[0]+fp[0]))
                        print('precision for all',(tp[0]+tp[1])/(tp[0]+tp[1]+fp[0]))

                        print('recall for visible',tp[0]/(tp[0]+fn[0]))
                        print('recall for all',(tp[0]+tp[1])/(tp[0]+tp[1]+fn[1]))
                        return((2*tp[0])/(2*tp[0]+fp[0]+fn[0]),tp[0]/(tp[0]+fp[0]),tp[0]/(tp[0]+fn[0]))
    except Exception as inst:
         print(inst)
         return((2*tp[0])/(2*tp[0]+fp[0]+fn[0]),1,1)


def score_in_treshold_function(path,actor,model):
    pr = []
    f1s = []
    condition = lambda x: x.startswith('B04'+actor)
    mean,std,treshold_min,treshold_max = find_treshold(model,path+"\\histogram.png",condition)
    tresholds= np.linspace(treshold_min,treshold_max+std,13)[3:]
    for treshold in tresholds:
        f1,prec,recall = score(model,treshold,condition)
        pr.append([prec,recall])
        f1s.append([f1,treshold])
        print(pr)
        print(f1s)
        print(treshold)
    pr.sort(key=lambda x: x[1])
    tosave = pd.DataFrame(pr)
    tosave.to_csv(path+"\\PR.csv",sep = ';',decimal = ',')
    precisions = [i[0] for i in pr ]
    recalls = [i[1] for i in pr ]
    plt.clf()
    plt.plot(recalls,precisions)
    plt.show()

    plt.clf()
    f1s.sort(key=lambda x: x[1])
    tosave = pd.DataFrame(f1s)
    tosave.to_csv(path+"\\F1.csv",sep = ';',decimal = ',')
    f1 = [i[0] for i in f1s ]
    tr = [i[1] for i in f1s]
    plt.plot(tr ,f1)
    plt.show()

# if __name__ == 'score':
path = ".\\Modele\\B04"

for actor in range(6,19):
    try:
        model = Load_model(path+n(actor))
        score_in_treshold_function(path+n(actor),n(actor),model)
    except :
        continue

