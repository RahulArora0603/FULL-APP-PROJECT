import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression , LogisticRegression
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

def main_Func():
    main_option = input("Press A to View Data, Press B for Data Visualization , Press C for Perform Machine Learning Operations, Press D to create own Csv, Press E to perform Statistical Operations")
    options = ["A","B","C","D","E","F","G","H"]
    funcs = [view_data]

def view_data():
    headings = list(df.head(0))
    view = input("Press A to view All Data , Press B to view Specific Column/ Multiple columns, Press C to view Grouped Data, :\n")
    view_list = ["A","B","C","D"]
    view_methods = [all_Data, specific_Column, grouped_Data]
    if view in view_list:
        r = view_list.index(view)
        view_methods[r]()

    def all_Data():
        print(df)

    def specific_Column():
        col_name = input("Enter column/columns : ")
        if ',' in col_name:
            col_list = col_name.split(',')
            column_list = []
            for col in col_list:
               if col in headings:
                   column_list.append(col)
                   continue
               else:
                   print(f"No column(s) {col} found")       
            print(df[column_list])
        else:
            print(df[col_name])

    def grouped_Data():
        group_by_col = input("Group according to: ")
        if group_by_col in headings:
            print(df.groupby(group_by_col))
        else:
            print("No such column found")

def data_Visualisation():
    graphtype = input("Press A for Line Chart , Press B for Bar Graph , Press C for Pie Chart, Press D for Histogram , Press E for Scatter Plot, Press G for Main Menu, Press H to exit \n")
    grp_options = ["A","B","C","D","E","F", "G","H"]
    def line_Chart():
        col_name1 = input("Enter first column(numerical values): \n")
        col_name2 = input("Enter second column(numerical values): \n")
        grp_title = input("Enter graph title : ") 
        x , y = df[col_name1].values , df[col_name2].values
        plt.plot(x , y, 'bo')
        plt.title(grp_title)
        plt.xlabel(col_name1)
        plt.ylabel(col_name2)

    def bar_Graph():
        category = input("Enter categorical heading : ")
        grp_title = input("Enter graph title : ")
        bar_color = input("Enter bar color : ")
        x = df[category].value_counts().index
        y = df[category].value_counts().values
        plt.bar(x, y, color=bar_color)
        plt.title(grp_title)
        plt.show()

    def pie_Chart():
        category = input("Enter categorical heading : ")
        grp_title = input("Enter graph title : ")
        x = df[category].value_counts().index
        y = df[category].value_counts().values
        fig , ax = plt.subplots()
        ax.pie(y, labels=x)
        plt.show()

    def histoGram():
        print("Warning : Use histogram only when the data is continuous. Example : Marks of Students")
        col_name = input("Enter column : ")
        no_bins = int(input("Number of bins : "))
        x = df[col_name].values
        plt.hist(x ,bins=no_bins)
        plt.show()

    def scatter_Plot():
        sc_col1 = input("Enter x-axis column: ")
        sc_col2 = input("Enter y-axis column: ")
        sc_color = input("Enter color of points: ")
        x = df[sc_col1]
        y = df[sc_col2]
        plt.scatter(x, y , c=sc_color)
        plt.xlabel(sc_col1)
        plt.ylabel(sc_col2)
        plt.show()

def mach_Learn():
    modtype = input("Press A for KNNeighbours Prediction , Press B for Random Forest Classifier, Press C for Decision Tree, Press 'D' for Linear Regression , 'E' for Logistic Regression, 'F' to return to main menu : ")
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
       else:
         mypred = int(to_predict)
         prediction = model.predict([[mypred]])
         print(prediction)
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
       else:
         mypred = int(to_predict)
         prediction = model.predict([[mypred]])
         print(prediction)

def checkDataType(col_list):
    df[]
