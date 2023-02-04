import pandas as pd
import os
PATH = "\\Dataset\\Caren\\Connected\\normalized\\"
files = list()
for filename in os.listdir(os.getcwd()+PATH):
        if filename.endswith(".xlsx"):
            file =pd.read_excel(os.getcwd()+PATH+"\\"+filename,"sheet1")
            name = filename.split('.')[0]
            if int(name.split("B04")[1].split('_')[0]) <13:
                continue
            full_name = os.getcwd()+PATH+name+".csv"
            file.to_csv(full_name) # to_excel(writer,"sheet1",index=False)