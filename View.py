# import sys
# sys.path.append('AMCParser')
# import amc_parser as amc
# from Viewer import Viewer

# print(dir(Viewer))
# m = amc.parse_amc("Dataset\\07_01.amc")
# f = amc.parse_asf("Dataset\\12_01.asf")
# v=Viewer(f,m)
# v.run()
from matplotlib import pyplot as plt
import numpy as np

x = np.linspace(0,7*np.pi,1000)
z= [0]*1000
y=(np.sin(3*x)*(np.sin(x)+ np.sin(2*x))/2)
num_anomaly = 3
for i in range(num_anomaly):
    index = np.random.randint(0,1000)
    value = np.random.uniform(0,0.5)
    z = [z[j]+ value*np.exp(-np.abs(index-j)/2) for j in range(len(y))]
    y = [y[j] + value*np.exp(-np.abs(index-j)/2) for j in range(len(y))]
plt.plot(x,y)
plt.plot(x,z)
plt.show()
