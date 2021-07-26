import pandas as pd
from scipy import stats
import numpy as np
#this entire code is just to convert the .data file into a csv

data_file = "breast-cancer.data"
data = open("breast-cancer.data")


#declaring a dataframe
column_names = ["Sample code number","Clump Thickness","Uniformity of Cell Size","Uniformity of Cell Shape","Marginal Adhesion",
"Single Epithelial Cell Size","Bare Nuclei","Bland Chromatin","Normal Nucleoli","Mitoses","Class"]
df = pd.DataFrame(columns=column_names)

cols = dict() #creating a dictionary with each column name as a key
for i in column_names:
    cols[i] = []

for i in data.readlines(): #reading line by line
    line = i.split(",")
    for i in range(0,len(line)):
        if(line[i]!="?"):
            line[i] = int(line[i]) #converting the string to an integer
        else:
            line[i] = 999 #999 means ?
    
    for i in range(11):
        cols[column_names[i]].append(line[i])


for i in column_names: #add all these to the dataframe now
    df[i] = cols[i]

#mode
print(df.mode()["Bare Nuclei"][0])

df["Bare Nuclei"] = df["Bare Nuclei"].replace(999,df.mode()["Bare Nuclei"][0])

#dataframe is ready now
df.to_csv("breast_cancer_mode_replaced.csv")

        