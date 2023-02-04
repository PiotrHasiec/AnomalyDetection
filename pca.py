import numpy as np
import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import tensorflow as tf
files = os.listdir("Dataset\\Caren\\Connected")
x =[]
y = []
points=[]
anomalies = []
with open("model.json", 'r') as plik:
        loaded_model_json = plik.read()
i = 0
model = tf.keras.models.model_from_json(loaded_model_json)
for file_name in files:
        if i > 0: break
        i+=1
        try:
            data = pd.read_csv("Dataset\\Caren\\Connected\\"+file_name).get(['sLAnkleAngles','cLAnkleAngles','pLAnkleAngles','sLHipAngles','cLHipAngles','pLHipAngles','sLKneeAngles','cLKneeAngles','pLKneeAngles','sLWristAngles','cLWristAngles','pLWristAngles','sRAnkleAngles','cRAnkleAngles','pRAnkleAngles','sRHipAngles','cRHipAngles','pRHipAngles','sRKneeAngles','cRKneeAngles','pRKneeAngles','sRWristAngles','cRWristAngles','pRWristAngles'])           
            
            predicted_data = model.predict(data.to_numpy().reshape(data.shape[0],8,3))

            all_errors = (data.to_numpy().reshape(data.shape[0],8,3) - predicted_data)**2
            errors = [np.sqrt(np.sum(err)) for err in all_errors]
            anomalies += [d for i,d in enumerate(data.to_numpy()) if errors[i]]
            points += list(data.to_numpy())
        except:
            continue
pca = PCA(n_components=2)
tsne = TSNE(n_components=2)
# points = pca.fit_transform(points)
sample = [p for p in points if np.random.uniform()>0.1]
points = pca.fit_transform(np.array(sample))
# anomalies_points = pca.transform(np.array(anomalies))
x = [p[0] for p in points]
y = [p[1] for p in points]

# xa = [p[0] for p in anomalies_points]
# ya = [p[1] for p in anomalies_points]

plt.scatter(x,y)
# plt.scatter(xa,ya)
plt.show()