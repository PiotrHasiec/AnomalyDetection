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
        if os.path.isfile(path+'\\'+file) and file.endswith('.csv'):
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
    return model

    
# if __name__ == 'score':

def score(model):

    joints = ['sLAnkleAngles','cLAnkleAngles','pLAnkleAngles','sLHipAngles','cLHipAngles','pLHipAngles','sLKneeAngles','sLWristAngles','cLWristAngles','pLWristAngles','sRAnkleAngles','cRAnkleAngles','pRAnkleAngles','sRHipAngles','cRHipAngles','pRHipAngles','sRKneeAngles','sRWristAngles','cRWristAngles','pRWristAngles']
    # joints = ['sLAnkleAngles','cLAnkleAngles','pLAnkleAngles','sLHipAngles','cLHipAngles','pLHipAngles','sLKneeAngles','cLKneeAngles','pLKneeAngles','sLWristAngles','cLWristAngles','pLWristAngles','sRAnkleAngles','cRAnkleAngles','pRAnkleAngles','sRHipAngles','cRHipAngles','pRHipAngles','sRKneeAngles','cRKneeAngles','pRKneeAngles','sRWristAngles','cRWristAngles','pRWristAngles']

    # joints = ['sLAnkleAngles','cLAnkleAngles','pLAnkleAngles','sLHipAngles','cLHipAngles','pLHipAngles','sLKneeAngles','cLKneeAngles','pLKneeAngles','sRAnkleAngles','cRAnkleAngles','pRAnkleAngles','sRHipAngles','cRHipAngles','pRHipAngles','sRKneeAngles','cRKneeAngles','pRKneeAngles']

    
    # model = Model.LSTMAutoencoder(NUMBER_OF_TIMESTEPS_IN_SERIE,len(joints),int(19*NUMBER_OF_TIMESTEPS_IN_SERIE))
    model.build((100,NUMBER_OF_TIMESTEPS_IN_SERIE,len(joints)))
    model.compile(loss =tf.losses.mean_squared_error,optimizer= tf.keras.optimizers.Adadelta(learning_rate=1))
    model.load_weights(".\\Modele\\M5\\prediction_model_t+1_49.h5")


    files = os.listdir("Dataset\\Caren\\Connected\\normalized\\Train\\")
    np.random.shuffle(files)



    datas = [pd.read_csv("Dataset\\Caren\\Connected\\normalized\\Train\\"+file_name).get(joints) for file_name in files]
    datas = [data for data in datas if not data is None]
    train_y,test_y = train_test_split(datas, random_state = 1237)
    timestep_mean_errors = []
    for i,data in enumerate(test_y) :
        if True:#filename.endswith('.csv'):
            stds =[]
            means = []
            precs = []
    
            try:
                    # data = pd.read_csv("Dataset\\Caren\\Connected\\normalized\\Train\\"+filename ).get(joints).to_numpy()
                    data = data.to_numpy()
                
                    timeserie_train_datas = timeseries_dataset_from_array(data[:-1],data[1:],sequence_stride = 100,sequence_length=100,batch_size=1,shuffle=True,seed=1237)


                    print(files[i])


                    predicted_data = model.predict(timeserie_train_datas,verbose=0).reshape((-1,len(joints)))
                    dataByTimesteps_y = np.array(data[1:predicted_data.shape[0]+1])
                    all_errors = (dataByTimesteps_y - predicted_data)**2
                    errors = [np.sqrt(np.sum(err)) for err in all_errors]
                    timestep_mean_errors+=errors
                    
                    # std = np.std(errors)
                    # mean = np.mean(errors)
                    # prec = np.percentile(errors,90)
                    # stds.append(std)
                    # means.append(mean)
                    # precs.append(prec)

            except Exception as inst:
                    print(type(inst))    # the exception instance
                    print(inst.args)     # arguments stored in .args
                    print(inst)          # __str__ allows args to be printed directly,


    counts, bins = np.histogram(timestep_mean_errors,30)

    plt.stairs(counts, bins)
    plt.show()

    mean = np.mean(timestep_mean_errors)
    std = np.std(timestep_mean_errors)
    prec =np.percentile(timestep_mean_errors,99)
    print(mean,std,prec)
            

    for filename in os.listdir("Dataset\\Caren\\Connected\\normalized\\Annomaly\\") :
        if filename.endswith('.csv'):
            try:   
                    tp = [0,0]
                    fp = [0,0]
                    fn = [0,0]
                    print(filename)
                    data = pd.read_csv("Dataset\\Caren\\Connected\\normalized\\Annomaly\\"+filename ).get(joints)
                    data = data.get(joints)
                    timeserie_train_datas = timeseries_dataset_from_array(data[:-1],data[1:],sequence_stride = 100,sequence_length=100,batch_size=1,shuffle=True,seed=1237)
                    predicted_data = model.predict(timeserie_train_datas,verbose=0).reshape((-1,len(joints)))
                    

                    dataByTimesteps_y = np.array(data[1:predicted_data.shape[0]+1])
                    all_errors = (dataByTimesteps_y  - predicted_data)**2
                    errors = [np.sqrt(np.sum(err)) for err in all_errors]
                    # errors = [np.sqrt(np.sum(err)) for j in range(all_errors.shape[0]) for err in all_errors[j] ]

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
                        y.append((err)/mean)
                        x.append(j)
                        if  (err) > prec:#mean+2.57*std:
                                y_d.append((err)/mean)
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

                
                    
                    
                    plt.show()
                    try:
                        print('f1:',(2*tp[0])/(2*tp[0]+fp[0]+fn[0]))
                        print('f1:',(2*tp[1])/(2*tp[1]+fp[0]+fn[1]))

                        print('precision for visible',tp[0]/(tp[0]+fp[0]))
                        print('precision for all',(tp[0]+tp[1])/(tp[0]+tp[1]+fp[0]))

                        print('recall for visible',tp[0]/(tp[0]+fn[0]))
                        print('recall for all',(tp[0]+tp[1])/(tp[0]+tp[1]+fn[1]))
                    except Exception as inst:
                        print("")


            except Exception as inst:
                    print(type(inst))    # the exception instance
                    print(inst.args)     # arguments stored in .args
                    print(inst) 
model = Load_model()
score(model)