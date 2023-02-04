import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("B:\\Repos\\INZ_LSTM_AE\\Dataset\\Caren\\B0406\\Filmy i pliki tekstowe\\1057-data0001.txt",delimiter="\t")
time = [(t - data["Time"][0])*100 for t in data["Time"]]
plt.plot(time,data["Velo"])
plt.show()