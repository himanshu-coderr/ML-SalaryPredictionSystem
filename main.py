import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score      # to show performance

def welcome():
    print("Welcome to Salary Predicition System")
    print("Press Enter Key to proceed")
    input()

def checkcsv():
    csv_files = []                           # contain all the scv files
    cur_dir = os.getcwd()                    # it gets all the files from the current directory and store them in cur_dir
    content_list = os.listdir((cur_dir))     # it puts all the files in content_list
    for x in content_list:
        if x.split('.')[-1] == 'csv':
            csv_files.append(x)
    if(len(csv_files)) == 0:
        return "No csv file in the directory"
    else:
        return csv_files

def display_and_select_csv(csv_files):                      # all csv files are passed
    i = 0
    for file_name in csv_files:
        print(i,"....",file_name)                           # with i , file name is printed
        i+=1
    return  csv_files[int(input("Select file to create ML Model-->"))]    # here user will select the file (i.e -- i) and
                                                                       # the index corresponding to the csv_file will be returned and stored in csv_file


def graph(X_train, Y_train, regressionObject, X_test, Y_test, Y_pred):
    plt.scatter(X_train,Y_train,color='red',label='training data')
    plt.plot(X_train,regressionObject.predict(X_train),color='blue',label='Best fit')
    plt.scatter(X_test,Y_test,color='lightgreen',label='test data')
    plt.scatter(X_test,Y_pred,color='black',label='Pred test data')
    plt.title("Salary vs Experience")
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.legend()
    plt.show()

def main():
    welcome()
    try:
        csv_files = checkcsv()
        if csv_files == "No csv file in the directory":
            raise FileNotFoundError("No csv file in the directory")
        csv_file = display_and_select_csv(csv_files)              # get the csv file from the user and stored in csv_file
        print(csv_file,'is selected')
        print('Reading csv file')
        print('Creating Dataset')
        dataset = pd.read_csv(csv_file)      # this will generate the Dataframe using pandas
        print('Dataset Created')
        X = dataset.iloc[:,:-1].values      #all rows , first column is selected
        Y = dataset.iloc[:,-1].values       #all rows , second colummn is selected

        s = float(input("Enter test data size  (between 0 and 1 -> "))   # taking test data from user(%age of test data)  let say 0.2
        X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=s)   # values returned will be -- 80%,20%,80%,20%  resp.

        print('Model creation in progression')
        regressionObject = LinearRegression()   # making an object of LinearRegeression class
        regressionObject.fit(X_train,Y_train)   # calling an function and passing training data and now REGRESSION LINE is created
        print("Model is created")
        print("Press ENTER key to prdict test data in trained model")
        input()

        Y_pred = regressionObject.predict(X_test)   # testing data is passed

        i = 0
        print(X_test,'....',Y_test,' ....',Y_pred)
        while i<len(X_test):
            print(X_test[i],'....',Y_test[i],'....',Y_pred[i])
            i+=1
        print("Press Enter key to see above result in graphical format")
        input()
        graph(X_train, Y_train, regressionObject, X_test, Y_test , Y_pred)

        r2 = r2_score(Y_test,Y_pred)     # performance
        print("Our model is %2.2f%% accurate" %(r2*100))

        print("Now you can Predict Salary of an employee using our Model")
        print('\nEnter experiences in years of the candidates , separated by comma')

        exp = [float(e) for e in input().split(',')]
        ex = []
        for x in exp:
            ex.append([x])
        experience = np.array(ex)
        salaries = regressionObject.predict(experience)

        plt.scatter(experience,salaries,color='black')
        plt.xlabel('Years of Experience')
        plt.ylabel('Salaries')
        plt.show()

        d = pd.DataFrame({'Experience':exp,'Salaries':salaries})
        print(d)

    except FileNotFoundError:
        print("No csv file in the directory")
        print("Press ENTER key to exit")
        input()
        exit()

if __name__ == '__main__':
    main()
    input()
