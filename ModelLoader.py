import os
import pandas as pd
import tensorflow as tf
class ModelLoader():
    def Load_model(self, model_name,path = ".\\Modele\\M1"):
    
        name_weights = ''
        csv_data = None
        for file in os.listdir(path):
            if os.path.isfile(path+'\\'+file) and file.endswith('.csv') and file.startswith(model_name):
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
                try:
                    model.load_weights(path+'\\'+name+'.h5')
                except:
                    return model
        return model