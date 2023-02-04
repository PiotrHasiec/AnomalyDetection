import pandas as pd
import os
PATH = "\\Dataset\\Caren\\Connected"
files = list()
for filename in os.listdir(os.getcwd()+PATH):
        if filename.endswith(".csv"):
            file =pd.read_csv(os.getcwd()+PATH+"\\"+filename)
            name = filename.split('.')[0]
            if int(name.split("B04")[1].split('_')[0]) <13:
                continue
            full_name = os.getcwd()+"\\Dataset\\Caren\\Connected\\"+name+".xlsx"
            with  pd.ExcelWriter(full_name) as writer:
                file.to_excel(writer,"sheet1",index=False)