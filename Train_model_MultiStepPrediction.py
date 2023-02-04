from Model import *
from Data import load_time_serie
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  MinMaxScaler
from keras.utils import timeseries_dataset_from_array
a = list(range(20))

TIME_STEP = 1
EPOCHES = 500
NUMBER_OF_TIMESTEPS_IN_SERIE_IN = 700
NUMBER_OF_TIMESTEPS_IN_SERIE_OUT = 100
LATTEM_DIM = 64
PATH = ".\\Modele\\Mult_B04"
MODEL_NAME = "multiprediction_model_lr1_t"+str(TIME_STEP)
joints = ['sLAnkleAngles','cLAnkleAngles','pLAnkleAngles','sLHipAngles','cLHipAngles','pLHipAngles','sLKneeAngles','sLWristAngles','cLWristAngles','pLWristAngles','sRAnkleAngles','cRAnkleAngles','pRAnkleAngles','sRHipAngles','cRHipAngles','pRHipAngles','sRKneeAngles','sRWristAngles','cRWristAngles','pRWristAngles']
# joints = ['sLANK','cLANK','pLANK','sLASI','cLASI','pLASI','sLKNE','cLKNE','pLKNE','sLWRA','cLWRA','pLWRA','sRANK','cRANK','pRANK','sRASI','cRASI','pRASI','sRKNE','cRKNE','pRKNE','sRWRA','cRWRA','pRWRA']


def load_data(datas):
    timeserie_datas_x = [ timeseries_dataset_from_array(serie[:-TIME_STEP],None,sequence_stride = NUMBER_OF_TIMESTEPS_IN_SERIE_IN,sequence_length=NUMBER_OF_TIMESTEPS_IN_SERIE_IN,batch_size=1,shuffle=True,seed=1237,) for i,serie in enumerate(datas)]
    timeserie_datas_y = [ timeseries_dataset_from_array(serie[TIME_STEP:],None,sequence_stride = NUMBER_OF_TIMESTEPS_IN_SERIE_IN,sequence_length=NUMBER_OF_TIMESTEPS_IN_SERIE_IN,batch_size=1,shuffle=True,seed=1237,) for i,serie in enumerate(datas)]
    x = []
    y = []
    for batch in zip(timeserie_datas_x, timeserie_datas_y):
        inputs, targets = batch
        for s in inputs:
            x.append(s)
        for s in targets:
            y.append(s)

    
    train_x,test_x = train_test_split(x, random_state = 1237)
    train_y,test_y = train_test_split(y, random_state = 1237)

    train_x,test_x = np.array(train_x).reshape(-1,NUMBER_OF_TIMESTEPS_IN_SERIE_IN,len(joints)),np.array(test_x).reshape(-1,NUMBER_OF_TIMESTEPS_IN_SERIE_IN,len(joints))
    train_y,test_y = np.array(train_y).reshape(-1,NUMBER_OF_TIMESTEPS_IN_SERIE_IN,len(joints)),np.array(test_y).reshape(-1,NUMBER_OF_TIMESTEPS_IN_SERIE_IN,len(joints))
    return train_x,test_x,train_y,test_y

def n(number):
    if number<10:
        return '0'+str(number)
    return str(number)

if __name__ == '__main__':
    for actor in range(6,20):
        os.makedirs('.\\Modele\\B04'+n(actor), exist_ok=True)

        model = SingleStepLSTM(LATTEM_DIM,len(joints),return_sequences=True).model
        model.build((100,NUMBER_OF_TIMESTEPS_IN_SERIE_IN,len(joints)))
        # model.load_weights("prediction_model49.h5")
        model.summary()
        model.compile(loss =tf.losses.mean_squared_error,optimizer= tf.keras.optimizers.Adadelta(learning_rate=1))

        tf.keras.utils.plot_model(model,show_shapes=True)
        files = os.listdir("Dataset\\Caren\\Connected")#\\normalized\\Train\\

        datas = [pd.read_csv("Dataset\\Caren\\Connected\\"+file_name).get(joints) for file_name in files if file_name.startswith('B04'+n(actor))]#normalized\\Train\\
        datas = [data for data in datas if not data is None]
        if len(datas) == 0:
            continue
        # get
        train_x,test_x,train_y,test_y = load_data(datas)
        losses = []
        for ep in range(EPOCHES):
            evaluation = []
            loss = []
            history = model.fit(train_x, train_y)
            loss.append(history.history['loss'])
            evaluation.append(model.evaluate(test_x,test_y,verbose = 0))
            losses.append([np.mean(loss),np.mean(evaluation)])
            print(np.mean(loss),np.mean(evaluation))
            model_json = model.to_json()
            with open('.\\Modele\\B04'+n(actor)+"\\"+MODEL_NAME+"ldim"+str(LATTEM_DIM)+".json", "w") as json_file:
                json_file.write(model_json)


        # serialize weights to HDF5
            model.save_weights('.\\Modele\\B04'+n(actor)+"\\"+MODEL_NAME+"ldim"+str(LATTEM_DIM)+"_"+str(ep)+".h5")

        losses= pd.DataFrame(np.array(losses))
        losses.to_csv('.\\Modele\\B04'+n(actor)+"\\"+MODEL_NAME+"ldim"+str(LATTEM_DIM)+".csv",sep=';',decimal=',')

        model_json = model.to_json()
        with open('.\\Modele\\B04'+n(actor)+"\\"+MODEL_NAME+"ldim"+str(LATTEM_DIM)+".json", "w") as json_file:
            json_file.write(model_json)

   

