import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns' , 500)
#pd.set_option('display.max_rows' , 500)
def cleanColumn(column):
    skew = abs(df[column].skew())
    if skew> 1:
        df.fillna({column: df[column].median()}, inplace=True)
    else:
        df.fillna({column: df[column].mean()}, inplace=True)
def cleanByMode(column):
    df[column] = df[column].fillna(df[column].mode()[0])
def showSummaryValuesColumn(column):
    print(f'set of values of column {column} is : {set(df[df[column].notna()][column])}')
def plotColumn(column):
    plt.figure(figsize=(6,4))
    plt.boxplot(df[column])
    plt.title(f'plotting of {column}')
    plt.ylabel(column)
    plt.grid(1)
    plt.show()
def removeOutliers(df , column):
    q1 = df[column].quantile(.25)
    q3 = df[column].quantile(.75)
    IQR = q3 - q1
    lower = q1 - IQR * 1.5
    upper = q3 + IQR * 1.5
    df = df[(df[column] >= lower) & (df[column] <= upper)  ].reset_index(drop = True)
    return df 
df = pd.read_csv('horse-colic.csv')
rows = df.shape[0]

#drop all columns that have more than 50% missed values
missing_percentage = 100 * df.isna().sum() / rows
threshold = 50 
dropped_columns = missing_percentage[missing_percentage >= threshold].index
df = df.drop(columns = [*dropped_columns , 'hospital_number' , 'pain_2'])
#cleaning and normalize phase
cleanByMode('surgery')
df["diagnosis"] = df["diagnosis"] - 1 
cleanColumn('temperature')
cleanColumn('pulse')
cleanColumn('respiratory_rate')
cleanByMode('temp_of_extremities')
cleanByMode('peripheral_pulse')
cleanByMode('mucus_membrane')
cleanByMode('mucoid_discharge')
cleanByMode('attitude')
cleanByMode('pain')
cleanByMode('peristalsis')
cleanByMode('abdominal_distension')
cleanByMode('nasogastric_tube')
cleanByMode('nasogastric_reflux')
cleanByMode('abdominal_muscle_tension')
cleanByMode('pulse_rate')
cleanColumn('temp_of_extremities_2')
cleanColumn('peripheral_pulse_2')#no outlier exist here
df = removeOutliers(df ,'respiratory_rate')
df = removeOutliers(df ,'pulse')
df = removeOutliers(df ,'temperature')
df = removeOutliers(df , 'temp_of_extremities_2')
df.to_csv("cleaned_data.csv") 
print("data has cleaned successfully")