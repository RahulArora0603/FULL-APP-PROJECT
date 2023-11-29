import pandas as pd
from graph_file import *
import matplotlib.pyplot as plt
#Takes CSV as input -
csvname = input("Enter csv path: ")
csvname = list(csvname)
for i in csvname:
    if i=="\\":
       r = csvname.index(i)
       #csvname.remove(i)
       csvname[r] = "/"
mycsv = ''.join(csvname)
print(mycsv)
df = pd.read_csv(mycsv)
#Prints the columns of the csv
print(df.head(0))
#Main function
def mainFunc():
    instructions = input("Press A to View Data , Press B to Describe Data, Press C for Visual Analysis ,\nPress D for Machine Learning , Press F to QUIT :\n")
    if instructions=='A':
       view = input("Press A to view All Data , Press B to view Specific Column , Press C to view multiple columns")
       if view=='A':
           print(df)
           mainFunc() #Recursion
       if view=='B':
           print(df.head(0))
           colname = input("\nEnter column name: ")
           print(df[colname])
           mainFunc()
       if view=='C':
           colnames = input("Enter columns seperated by comma : ")
           mylist = colnames.split(',')
           print(df[mylist])
           mainFunc()
    elif instructions=='B':
        print(df.describe())
        mainFunc()
    elif instructions=='C': #Visual Analysis
        checkDataType(df)
        graphtype = input("Press A for Line Chart , Press B for Bar Graph , Press C for Pie Chart : ")
        if graphtype=='A':
            col1 = input("Enter value of X axis : ")
            col2 = input("Enter value of Y axis : ")
            x = df[col1].values
            y = df[col2].values
            plt.plot(x , y , 'ro')
            plt.show()
            mainFunc()
        if graphtype=='B':
            MyBarGraph(df)
            mainFunc()
        if graphtype=='C':
            print(df.head(0))
            myPieChart(df)
            mainFunc()
    elif instructions=='D': #Machine Learning for predictions 
        print("Choose the Machine Learning model based on Data :\n[I recommend checking if the data is linear using Graphs]\n")
        print(f"{df.head(0)}\n")
        modtype = input("Press 'D' for Linear Regression , 'E' for Logistic Regression, 'F' to return to main menu : ")
        if modtype=='D': #Linear Regression
           para1 = input("Enter independent variables : ")
           para2 = input("Enter dependent variables : ")
           comma = ','
           if comma in list(para1):
              para1 = para1.split(',')
              x = df[para1].values
              y = df[para2].values
           else:
              x = df[para1].values
              y = df[para2].values
              x = x.reshape(-1 , 1)
           X_train , X_test , Y_train , Y_test = train_test_split(x , y , test_size=1/4 , random_state=0)  
           model = LinearRegression()
           model.fit(X_train, Y_train)
           ypred = model.predict(X_test)
           print(ypred)
           to_predict = input("Enter values to predict: ")
           if comma in list(to_predict):
             to_predict=to_predict.split(",")
             ipredict = []
             for i in range(0, len(para1)):
                if str(df[para1[i]].dtype).startswith("int")==True:
                    r = int(to_predict[i])
                    ipredict.append(r)
                elif str(df[para1[i]].dtype).startswith("float")==True:
                    r = float(to_predict[i])
                    ipredict.append(r)
             print(ipredict)         
             prediction = model.predict([ipredict])
             print(prediction)
             mainFunc()   
           else:
             mypred = int(to_predict)
             prediction = model.predict([[mypred]])
             print(prediction)
             mainFunc()      
        elif modtype=='E': #Logistic Regression   
           para1 = input("Enter independent variables : ")
           para2 = input("Enter dependent variables : ")
           comma = ','
           if comma in list(para1):
              para1 = para1.split(',')
              x = df[para1].values
              y = df[para2].values
           else:
              x = df[para1].values
              y = df[para2].values
              x = x.reshape(-1 , 1)
           X_train , X_test , Y_train , Y_test = train_test_split(x , y , test_size=1/4 , random_state=0)  
           model = LogisticRegression()
           model.fit(X_train, Y_train)
           ypred = model.predict(X_test)
           print(ypred)
           to_predict = input("Enter values to predict: ")
           if comma in list(to_predict):
             to_predict=to_predict.split(",")
             ipredict = []
             for i in range(0, len(para1)):
                if str(df[para1[i]].dtype).startswith("int")==True:
                    r = int(to_predict[i])
                    ipredict.append(r)
                elif str(df[para1[i]].dtype).startswith("float")==True:
                    r = float(to_predict[i])
                    ipredict.append(r)
             print(ipredict)         
             prediction = model.predict([ipredict])
             print(prediction)
             mainFunc()
           else:
             mypred = int(to_predict)
             prediction = model.predict([[mypred]])
             print(prediction)
             mainFunc()
        elif modtype=='F': #Directed to Main Function 
           mainFunc()
    elif instructions=='F': #Quit
        print("Thank You for using this application🙏.Have a nice day😊")

mainFunc()