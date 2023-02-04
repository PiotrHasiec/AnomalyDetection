from Model import *

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from DataLoader import DataLoader
# from keras.utils import timeseries_dataset_from_array
a = list(range(20))

from functions import root_percentage_mean_error


TIME_STEP = 1
EPOCHES = 50
NUMBER_OF_TIMESTEPS_IN_SERIE = 100
laten_dim = 5
noise = 0.5

MODEL_NAME = "AE_model"
joints = ['sLAnkleAngles','cLAnkleAngles','pLAnkleAngles','sLHipAngles','cLHipAngles','pLHipAngles','sLKneeAngles','sLWristAngles','cLWristAngles','pLWristAngles','sRAnkleAngles','cRAnkleAngles','pRAnkleAngles','sRHipAngles','cRHipAngles','pRHipAngles','sRKneeAngles','sRWristAngles','cRWristAngles','pRWristAngles']
# joints = ['sLANK','cLANK','pLANK','sLASI','cLASI','pLASI','sLKNE','cLKNE','pLKNE','sLWRA','cLWRA','pLWRA','sRANK','cRANK','pRANK','sRASI','cRASI','pRASI','sRKNE','cRKNE','pRKNE','sRWRA','cRWRA','pRWRA']


# def load_data(datas):
#     timeserie_datas_x = [ timeseries_dataset_from_array(serie[:-TIME_STEP],None,sequence_stride = NUMBER_OF_TIMESTEPS_IN_SERIE,sequence_length=NUMBER_OF_TIMESTEPS_IN_SERIE,batch_size=1,shuffle=True,seed=1237,) for i,serie in enumerate(datas)]
#     timeserie_datas_y = [ timeseries_dataset_from_array(serie[TIME_STEP:],None,sequence_stride = NUMBER_OF_TIMESTEPS_IN_SERIE,sequence_length=NUMBER_OF_TIMESTEPS_IN_SERIE,batch_size=1,shuffle=True,seed=1237,) for i,serie in enumerate(datas)]
#     x = []
#     y = []
#     for batch in zip(timeserie_datas_x, timeserie_datas_y):
#         inputs, targets = batch
#         for s in inputs:
#             x.append(s)
#         for s in targets:
#             y.append(s)

    
#     train_x,test_x = train_test_split(x, random_state = 1237)
#     train_y,test_y = train_test_split(y, random_state = 1237)

#     train_x,test_x = np.array(train_x).reshape(-1,NUMBER_OF_TIMESTEPS_IN_SERIE,len(joints)),np.array(test_x).reshape(-1,NUMBER_OF_TIMESTEPS_IN_SERIE,len(joints))
#     train_y,test_y = np.array(train_y).reshape(-1,NUMBER_OF_TIMESTEPS_IN_SERIE,len(joints)),np.array(test_y).reshape(-1,NUMBER_OF_TIMESTEPS_IN_SERIE,len(joints))
#     return train_x,test_x,train_y,test_y



if __name__ == '__main__':
    files = os.listdir("Dataset\\Caren\\Connected\\normalized\\Train\\")#

    datas = [pd.read_csv("Dataset\\Caren\\Connected\\normalized\\Train\\"+file_name).get(joints) for file_name in files]#normalized\\Train\\
    datas = [data for data in datas if not data is None]
    loader = DataLoader()
    train_x,test_x,train_y,test_y = loader.load_data(datas,1,NUMBER_OF_TIMESTEPS_IN_SERIE,NUMBER_OF_TIMESTEPS_IN_SERIE)
    i = 1
    for laten_dim in [50]:
        for noise in [0.7]:
            PATH = '.\\Modele\AE_ldim_{}_noise_{}_timsetps_{}'.format(laten_dim,noise,NUMBER_OF_TIMESTEPS_IN_SERIE)
            os.makedirs(PATH, exist_ok=True)

            model = Autoencoder((NUMBER_OF_TIMESTEPS_IN_SERIE,len(joints)),laten_dim,noise).model
            try:
                model.load_weights(PATH+"\\AE_model_ldim_50_noise_0.7_epoch_45.h5")
            except:
                print("aa")
            model.build((1,NUMBER_OF_TIMESTEPS_IN_SERIE,len(joints)))
            model.summary()
            model.compile(loss =tf.losses.mean_squared_error,optimizer= tf.keras.optimizers.Adadelta(learning_rate=1),metrics=[root_percentage_mean_error])

            tf.keras.utils.plot_model(model,PATH+'\\{}_ldim_{}_noise_{}.png'.format(MODEL_NAME,laten_dim,noise),show_shapes=True)
            losses = []
            model_json = model.to_json()
            with open(PATH+'\\{}_ldim_{}_noise_{}.json'.format(MODEL_NAME,laten_dim,noise), "w") as json_file:
                    json_file.write(model_json)
            for ep in range(EPOCHES):
                evaluation = []
                loss = []
                history = model.fit(train_x, train_x,batch_size=16, epochs=10)
                loss.append(history.history['loss'])
                evaluation.append(model.evaluate(test_x,test_x,verbose = 0)[0])
                losses.append([np.mean(loss),np.mean(evaluation)])
                print(np.mean(loss),np.mean(evaluation))
                try:
                    if ep> 1 and min(losses[ep][1],losses[ep-1][1])>losses[ep-2][1]:
                        break
                except:
                    print('źle napisałeś kod')
                
                model.save_weights(PATH+'\\{}_ldim_{}_noise_{}_epoch_{}.h5'.format(MODEL_NAME,laten_dim,noise,ep))

            losses= pd.DataFrame(np.array(losses))
            losses.to_csv(PATH+'\\{}_ldim_{}_noise_{}.csv'.format(MODEL_NAME,laten_dim,noise),sep=';',decimal=',')

            model_json = model.to_json()
            with open(PATH+'\\{}_ldim_{}_noise_{}.json'.format(MODEL_NAME,laten_dim,noise), "w") as json_file:
                json_file.write(model_json)

   

