#from prepare_data import df
from Model import LogisticRegression
from numpy import  array
import pandas as pd
from utils import confusionMatrix

df = pd.read_csv("cleaned_data.csv")
length = int(len(df) * 0.7)
#split data for training phase 
data = df.loc[:length - 1:]
y = array(data["diagnosis"])
data = data.drop(columns = ["diagnosis"])
data = array(data)
model = LogisticRegression()
model.fit(X = data , y = y)
predict = array(model.predict(X = data))
confusionMatrix(y , predict , 'logistic report.txt' , "train")
y = array(df.loc[length :: ,"diagnosis"])
df = df.drop(columns = ["diagnosis"])
data = array(df.iloc[length ::])
predict = array(model.predict(X = data))
confusionMatrix(y , predict , 'logistic report.txt')
