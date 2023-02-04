import numpy as np

from sklearn.model_selection import train_test_split
from keras.utils import timeseries_dataset_from_array

NUMBER_OF_TIMESTEPS_IN_SERIE_IN = 700
NUMBER_OF_TIMESTEPS_IN_SERIE_OUT = 100
TIME_STEP = 10
JOINTS = ['sLAnkleAngles','cLAnkleAngles','pLAnkleAngles','sLHipAngles','cLHipAngles','pLHipAngles','sLKneeAngles','sLWristAngles','cLWristAngles','pLWristAngles','sRAnkleAngles','cRAnkleAngles','pRAnkleAngles','sRHipAngles','cRHipAngles','pRHipAngles','sRKneeAngles','sRWristAngles','cRWristAngles','pRWristAngles']
  
class DataLoader():
    def __init__(self,joints =JOINTS,test_size = 0.2 ):
        self.joints = joints
        self.test_size = test_size

    def load_data(self,datas,timestep = TIME_STEP,number_of_timesteps_in = NUMBER_OF_TIMESTEPS_IN_SERIE_IN,number_of_timesteps_out=NUMBER_OF_TIMESTEPS_IN_SERIE_OUT):
        if timestep > 0:
            timeserie_datas_x = [ timeseries_dataset_from_array(serie[:-timestep],None,sequence_stride = number_of_timesteps_in,sequence_length=number_of_timesteps_in,batch_size=1,shuffle=True,seed=1237,) for i,serie in enumerate(datas)]
        else:
            timeserie_datas_x = [ timeseries_dataset_from_array(serie.get(JOINTS),None,sequence_stride = number_of_timesteps_in,sequence_length=number_of_timesteps_in,batch_size=1,shuffle=True,seed=1237,) for i,serie in enumerate(datas)]
        timeserie_datas_y = [ timeseries_dataset_from_array(serie.get(JOINTS)[timestep:],None,sequence_stride = number_of_timesteps_in,sequence_length=number_of_timesteps_out,batch_size=1,shuffle=True,seed=1237,) for i,serie in enumerate(datas)]
        x = []
        y = []

        for batch in zip(timeserie_datas_x, timeserie_datas_y):
            inputs, targets = batch
            for s in inputs:
                x.append(s)
            for s in targets:
                y.append(s)
        if self.test_size >0:
            train_x,test_x = train_test_split(x, random_state = 1237,test_size= self.test_size)
            train_y,test_y = train_test_split(y, random_state = 1237,test_size= self.test_size)

            train_x,test_x = np.array(train_x).reshape(-1,number_of_timesteps_in,len(self.joints)),np.array(test_x).reshape(-1,number_of_timesteps_in,len(self.joints))
            train_y,test_y = np.array(train_y).reshape(-1,number_of_timesteps_out,len(self.joints)),np.array(test_y).reshape(-1,number_of_timesteps_out,len(self.joints))
            return train_x,test_x,train_y,test_y
        else:
            train_x,test_x = np.array(x).reshape(-1,number_of_timesteps_in,len(self.joints)),None
            train_y,test_y = np.array(y).reshape(-1,number_of_timesteps_out,len(self.joints)),None
            return train_x,test_x,train_y,test_y
    def production_load_data(self,data,timestep):
        data = data.get(self.joints)
        x = np.array(data)[:(data.shape[0]//timestep)*timestep]
        x =x.reshape(-1,timestep,len(self.joints))
        return x