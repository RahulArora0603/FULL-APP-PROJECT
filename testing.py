import pandas as pd
import matplotlib.pyplot as plt

'''df = pd.read_csv('bmi.csv')

x = df['Age'].value_counts()
y = df['Age'].value_counts().keys()
plt.bar(y ,x , color ="black")
plt.show()'''

df = [180,313,101,255,202,198,109,183,181,113,171,165,318,145,131,145,226,113,268,108]
df1 = pd.Series(df)
print(df1.mean())
print(df1.std(ddof=0))