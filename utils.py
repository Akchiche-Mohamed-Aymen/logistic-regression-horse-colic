def confusionMatrix(y , predict ,fname ,  phase = 'testing' ):
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
    #create report
    with open(fname , 'a' if phase == 'testing' else 'w') as f:
        try:
            accuracy = round((confusion_matrix[0] + confusion_matrix[3])/length , 3)* 100
            precision = round(confusion_matrix[0]/(confusion_matrix[0] + confusion_matrix[2]) , 3)* 100
            recall = round(confusion_matrix[0]/(confusion_matrix[0] + confusion_matrix[1]) , 3)* 100
            f.write(f"\n {'**' * 15} for {phase} phase {'**' * 15}\n")        
            f.write(f'  Accuracy of {phase} is {accuracy}%. \n')
            f.write(f'  Precision of {phase} is {precision}%. \n')
            f.write(f'  Recall / Sensitivity / TPR  of {phase} is {recall}%. \n')
        except:
            print(f"error in {fname}")