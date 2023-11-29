import pandas as pd
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
    graphtype = input("Press A for Line Chart , Press B for Bar Graph , Press C for Pie Chart, Press D for Histogram , Press E for Scatter Plot, Press F for Area Chart, Press G for Main Menu, Press H to exit \n")
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
        x = 