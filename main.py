import numpy as np
from sklearn.neural_network import MLPClassifier
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

train_labels = []
with open("train_labels.txt") as f:
    for line in f:
        train_labels.append(int(line))


raw_data = []
with open("train_data.txt") as f:
    for line in f:
        line = line.split()
        temp = []
        if len(line) == 5:
            for n in line:
                temp.append(float(n))
        if temp != []:
            raw_data.append(temp)


train_data = []

for i in range(len(train_labels)):
    temp = []
    for j in range(7):
        for num in raw_data[i*7+j]:
            temp.append(num)
    train_data.append(temp)

train_data_array = np.array(train_data)

# train_data_array.reshape(-1,1)

classifier = MLPClassifier(solver='lbfgs', alpha=1e-6, hidden_layer_sizes=(3, 4), random_state=69)
classifier.fit(train_data_array, train_labels)




test_labels = []
with open("test_labels.txt") as f:
    for line in f:
        test_labels.append(int(line))


raw_data = []
with open("test_data.txt") as f:
    for line in f:
        line = line.split()
        temp = []
        if len(line) == 5:
            for n in line:
                temp.append(float(n))
        if temp != []:
            raw_data.append(temp)


test_data = []

for i in range(len(test_labels)):
    temp = []
    for j in range(7):
        for num in raw_data[i*7+j]:
            temp.append(num)
    test_data.append(temp)

test_data_array = np.array(test_data)

results = classifier.predict(test_data_array)

df = pd.DataFrame(data=[test_labels,results], columns=['y_Actual','y_Predicted'])
confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])

sn.heatmap(confusion_matrix, annot=True)
plt.show()

