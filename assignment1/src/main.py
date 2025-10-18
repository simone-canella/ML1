#!/usr/bin/env python3
import pandas as pd
from NBayesClassifier import Nbayes

DEBUG = False

'''
---------------------
TASK1: PREPARING DATA
---------------------
'''
#IMPORT DATA:
df_weather = pd.read_csv("../data/weather.data.csv", sep='\s+')

if DEBUG == True:
    print(df_weather.shape)


#CLEAN DATA:
df_weather.columns = df_weather.columns.str.replace('#', '') #substitute '#' with '' from index

if DEBUG == True:
    print("UPLOADED DATAFRAME: \n", df_weather.columns, "\n") #control that is done correctly


#SPLIT TARGET:
#x_train = df_weather.iloc[:, 0:(df_weather.columns.size - 1)] #select first 4 columns 
x_train = df_weather.iloc[:, 0 : -1] #select first 4 columns 

if DEBUG == True:
    print("TRAIN CONDITION: \n", x_train, "\n")

y_train = df_weather.iloc[:, -1] #select last column

if DEBUG == True:
    print("TRAIN EFFECT: \n", y_train, "\n")


#COMPUTE NUMBER OF LEVELS:
levels = list(range(len(x_train.columns))) #create a list that is big as the number of columns of x_train

for i in range(0, len(x_train.columns)):
    levels[i] = x_train.iloc[:, i].nunique() #count unique element of each column

if DEBUG == True:
    print("LEVELS FOR EACH CLASS: \n", levels, "\n")

'''
---------------------
TASK2.1: FIT METHOD
---------------------
'''

classifier = Nbayes()

classifier.fit(x_train, y_train)

classifier.predict(x_train)

