import pandas as pd
import numpy as np
import os
JOINTS = ['sLAnkleAngles','cLAnkleAngles','pLAnkleAngles','sLHipAngles','cLHipAngles','pLHipAngles','sLKneeAngles','cLKneeAngles','pLKneeAngles','sLWristAngles','cLWristAngles','pLWristAngles','sRAnkleAngles','cRAnkleAngles','pRAnkleAngles','sRHipAngles','cRHipAngles','pRHipAngles','sRKneeAngles','cRKneeAngles','pRKneeAngles','sRWristAngles','cRWristAngles','pRWristAngles']
class DataNormalizer():
    def __init__(self,path_norm,joints = JOINTS):
        self.joints = joints
        source_datas = [pd.read_csv(path_norm+file_name) for file_name in os.listdir(path_norm)]
        self.connected = pd.DataFrame()
        for data in source_datas:
            self.connected = pd.concat([data,self.connected])
        self.parameters ={col:(self.connected[col].max(),self.connected[col].min()) for col in self.connected.columns}

    def normalize(self,path):
        files = os.listdir(path)
        
        datas = [pd.read_csv(path+file_name) for file_name in os.listdir(path)]
        normalized_df =pd.DataFrame()
        for i,data in enumerate(datas):
            normalized_df =pd.DataFrame()
            for col in data.columns:
                if col in self.joints:
                    c = (2*(data[col] - self.parameters[col][1])/(self.parameters[col][0]-self.parameters[col][1]))-1
                elif not col in self.joints:
                    c = data[col]
                else:
                    continue
                normalized_df[col]=c
            normalized_df.to_csv(path+files[i])


path_norm = "Dataset\\Caren\\Connected\\normalized\\train\\"
normalier = DataNormalizer(path_norm)

path = "Dataset\\Caren\\Connected\\normalized\\test\\"
normalier.normalize(path)

path = "Dataset\\Caren\\Connected\\normalized\\Annomaly\\"
normalier.normalize(path)

path = "Dataset\\Caren\\Connected\\normalized\\train\\"
normalier.normalize(path)
