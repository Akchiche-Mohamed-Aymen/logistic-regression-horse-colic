from Model import  K_nearest_neighbors
from numpy import  array
from utils import confusionMatrix
import pandas as pd

for k in range(3 , 8 , 2):
    df = pd.read_csv("cleaned_data.csv")
    length = int(len(df) * 0.7)
    #split data for training phase 
    data = df.loc[:length - 1:]
    y = array(data["diagnosis"])
    data = data.drop(columns = ["diagnosis"])
    data = array(data)

    model = K_nearest_neighbors(nb_neighbors=k)
    model.fit(X = data , Y = y)
    predict = [model.predict(newX = row) for row in data]
    confusionMatrix(y , predict ,f'Knn report {k}.txt' ,  "train")
    y = array(df.loc[length :: ,"diagnosis"])
    df = df.drop(columns = ["diagnosis"])
    data = array(df.iloc[length ::])
    predict = [model.predict(newX = row) for row in data]
    confusionMatrix(y , predict ,f'Knn report {k}.txt')
