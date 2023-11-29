import pandas as pd
'''def main_Func():
    view1 = input("Press A to view All Data") 
    options = ["A","B","C","D","E","F","G","H"]
    funcs = [view_data]
    if view1 in options:
        r = options.index(view1)
        funcs[r]()

def view_data():
    view = input("Press A to view All Data , Press B to view Specific Column , Press C to view multiple columns, Press D to view Grouped Data, ")
    view_list = ["A","B","C","D"]
    if view=="A":
        print("Ok")

main_Func()'''
df = pd.read_csv('bmi.csv')
col = input("Enter column")
print(df[[col]])
