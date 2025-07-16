#from prepare_data import df
from Model import LogisticRegression
from numpy import  array
import pandas as pd

def confusionMatrix(y , predict , phase = 'testing'):
    confusion_matrix = [0] * 4
    length = len(y)
    for i in range(length):
        if y[i] == 1:
            if predict[i] == 1:
                confusion_matrix[0] +=1
            else:
                confusion_matrix[1] +=1
        else : 
            if predict[i] == 1:
                confusion_matrix[2] +=1
            else:
                confusion_matrix[3] +=1
    with open('withoutOutliers.txt' , 'a' if phase == 'testing' else 'w') as f:
        accuracy = round((confusion_matrix[0] + confusion_matrix[3])/length , 3)* 100
        precision = round(confusion_matrix[0]/(confusion_matrix[0] + confusion_matrix[2]) , 3)* 100
        recall = round(confusion_matrix[0]/(confusion_matrix[0] + confusion_matrix[1]) , 3)* 100
        f.write(f"\n {'**' * 30} \n")
        f.write(f'  Accuracy of {phase} is {accuracy}%. \n')
        f.write(f'  Precision of {phase} is {precision}%. \n')
        f.write(f'  Recall / Sensitivity / TPR  of {phase} is {recall}%. \n')

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
confusionMatrix(y , predict , "train")
y = array(df.loc[length :: ,"diagnosis"])
df = df.drop(columns = ["diagnosis"])
data = array(df.iloc[length ::])
predict = array(model.predict(X = data))
confusionMatrix(y , predict)
