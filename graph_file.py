import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler 
#import csv
df = pd.read_csv('./sk learn/HR_comma_sep.csv')
#df.to_csv(index=1, )
#print(type(df))
def MultiColInput():
   colnames = input("Enter columns seperated by comma : ")
   mylist = colnames.split(',')
   print(df[mylist])

def checkDataType(val):
   cols = val.head(0)
   headlist = []
   for i in cols:
      if str(val[i].dtype).startswith("int")==True or str(val[i].dtype).startswith("float")==True :
        headlist.append(i)
   print(headlist)

def MyBarGraph(val):
    category = input("Enter categorical heading : ")
    x = val[category].value_counts().index
    y = val[category].value_counts().values
    print(x)
    plt.bar(x, y)
    plt.show()

def myPieChart(val):
    category = input("Enter categorical heading : ")
    x = val[category].value_counts().index
    y = val[category].value_counts().values
    plt.pie(y , labels=x)
    plt.show()
