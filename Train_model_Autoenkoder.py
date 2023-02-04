from Model import *
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  MinMaxScaler
from keras.utils import timeseries_dataset_from_array
a = list(range(20))


def roor_percentage_mean_error(y_true, y_pred):
    err = tf.sqrt( tf.keras.metrics.mean_squared_error(y_true,y_pred)/tf.reduce_mean(tf.square(y_pred)))
    return err

TIME_STEP = 1
EPOCHES = 50
NUMBER_OF_TIMESTEPS_IN_SERIE = 100
LATTEM_DIM = 4
PATH = ".\\Modele\LSTM_04_AE"+str(LATTEM_DIM)+"_B04"
MODEL_NAME = "AE_model_t"+str(NUMBER_OF_TIMESTEPS_IN_SERIE)
joints = ['sLAnkleAngles','cLAnkleAngles','pLAnkleAngles','sLHipAngles','cLHipAngles','pLHipAngles','sLKneeAngles','sLWristAngles','cLWristAngles','pLWristAngles','sRAnkleAngles','cRAnkleAngles','pRAnkleAngles','sRHipAngles','cRHipAngles','pRHipAngles','sRKneeAngles','sRWristAngles','cRWristAngles','pRWristAngles']
# joints = ['sLANK','cLANK','pLANK','sLASI','cLASI','pLASI','sLKNE','cLKNE','pLKNE','sLWRA','cLWRA','pLWRA','sRANK','cRANK','pRANK','sRASI','cRASI','pRASI','sRKNE','cRKNE','pRKNE','sRWRA','cRWRA','pRWRA']


def load_data(datas):
    timeserie_datas_x = [ timeseries_dataset_from_array(serie[:-TIME_STEP],None,sequence_stride = NUMBER_OF_TIMESTEPS_IN_SERIE,sequence_length=NUMBER_OF_TIMESTEPS_IN_SERIE,batch_size=1,shuffle=True,seed=1237,) for i,serie in enumerate(datas)]
    timeserie_datas_y = [ timeseries_dataset_from_array(serie[TIME_STEP:],None,sequence_stride = NUMBER_OF_TIMESTEPS_IN_SERIE,sequence_length=NUMBER_OF_TIMESTEPS_IN_SERIE,batch_size=1,shuffle=True,seed=1237,) for i,serie in enumerate(datas)]
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

    train_x,test_x = np.array(train_x).reshape(-1,NUMBER_OF_TIMESTEPS_IN_SERIE,len(joints)),np.array(test_x).reshape(-1,NUMBER_OF_TIMESTEPS_IN_SERIE,len(joints))
    train_y,test_y = np.array(train_y).reshape(-1,NUMBER_OF_TIMESTEPS_IN_SERIE,len(joints)),np.array(test_y).reshape(-1,NUMBER_OF_TIMESTEPS_IN_SERIE,len(joints))
    return train_x,test_x,train_y,test_y

def n(number):
    if number<10:
        return '0'+str(number)
    return str(number)

if __name__ == '__main__':
    for actor in range(19,20):
        os.makedirs(PATH+n(actor), exist_ok=True)

        model = Autoencoder([NUMBER_OF_TIMESTEPS_IN_SERIE,len(joints)],LATTEM_DIM).model
        model.build((NUMBER_OF_TIMESTEPS_IN_SERIE,len(joints)))
        model.summary()
        model.compile(loss =tf.losses.mean_squared_error,optimizer= tf.keras.optimizers.Adadelta(learning_rate=1),metrics=[roor_percentage_mean_error])

        tf.keras.utils.plot_model(model,show_shapes=True)
        files = os.listdir("Dataset\\Caren\\Connected\\normalized\\Train\\")#

        datas = [pd.read_csv("Dataset\\Caren\\Connected\\normalized\\Train\\"+file_name).get(joints) for file_name in files if file_name.startswith('B04'+n(actor))]#normalized\\Train\\
        datas = [data for data in datas if not data is None]
        if len(datas) == 0:
            continue
        # get
        train_x,test_x,train_y,test_y = load_data(datas)
        losses = []
        for ep in range(EPOCHES):
            evaluation = []
            loss = []
            history = model.fit(train_x, train_x,batch_size=16, epochs=10)
            loss.append(history.history['loss'])
            evaluation.append(model.evaluate(test_x,test_x,verbose = 0)[0])
            losses.append([np.mean(loss),np.mean(evaluation)])
            print(np.mean(loss),np.mean(evaluation))
            model_json = model.to_json()
            with open(PATH+n(actor)+"\\"+MODEL_NAME+"ldim"+str(LATTEM_DIM)+".json", "w") as json_file:
                json_file.write(model_json)


        # serialize weights to HDF5
            model.save_weights(PATH+n(actor)+"\\"+MODEL_NAME+"ldim"+str(LATTEM_DIM)+"_"+str(ep)+".h5")

        losses= pd.DataFrame(np.array(losses))
        losses.to_csv(PATH+n(actor)+"\\"+MODEL_NAME+"ldim"+str(LATTEM_DIM)+".csv",sep=';',decimal=',')

        model_json = model.to_json()
        with open(PATH+n(actor)+"\\"+MODEL_NAME+"ldim"+str(LATTEM_DIM)+".json", "w") as json_file:
            json_file.write(model_json)

   

