import pandas as pd
import numpy as np
import os
PATH = "\\Dataset\\Caren\\B04"
def n(number):
    if number<10:
        return '0'+str(number)
    return str(number)

for i in range(6,13):
    files = list()
    for filename in os.listdir(os.getcwd()+PATH+n(i)+"\\Serie czasowe\\Markery"):
        if filename.endswith(".csv"):
            splited = filename.split(" ")
            serie = splited.pop().split(".")[0]
            joint = filename.split('_')[0]
            header = ['s'+joint,'c'+joint,'p'+joint,'other'+joint]
            file = pd.read_csv(os.getcwd()+PATH+n(i)+"\\Serie czasowe\\Markery\\"+filename,names=header)
            file = file.get(file.columns[0:3])
            if not (serie+"B04"+n(i) in files):
                files.append(serie+"B04"+n(i))
                for filename2 in os.listdir(os.getcwd()+PATH+n(i)+"\\Serie czasowe\\Markery"):
                    joint = filename2.split('_')[0]
                    header = ['s'+joint,'c'+joint,'p'+joint,'other'+joint]
                    if filename2.endswith(serie+".csv") and filename!=filename2:
                        file2 = pd.read_csv(os.getcwd()+PATH+n(i)+"\\Serie czasowe\\Markery\\"+filename2,names=header)
                        for col in  file2.columns[0:3]:
                            file = file.join(file2[col])

                file.to_csv(os.getcwd()+"\\Dataset\\Caren\\Connected\\"+"M04"+n(i)+"_"+serie+".csv",index=False)
            # files[serie+"B04"+n(i)] = file
            # f = files.get(serie+"B04"+n(i))
            # if not f  is None:
            #     file = f.merge(file)
            
            # print(os.getcwd()+PATH+n(i)+"\\Serie czasowe\\Markery\\"+filename)
    # for fkey in files.keys():
    #     files[fkey].to_csv(os.getcwd()+"\\Dataset\\Caren\\Connected\\"+fkey+".csv")
    # print(files)
