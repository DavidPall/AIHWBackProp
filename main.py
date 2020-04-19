from sklearn.neural_network import MLPClassifier
import numpy as np

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
        raw_data.append(temp)

train_data = []

for i in range(len(raw_data)//7):
    temp = []
    for j in range(7):
        for num in raw_data[i*7+j]:
            temp.append(num)
    train_data.append(temp)


train_data_array = np.array(train_data)
train_data_array.reshape(-1,1)

classifier = MLPClassifier(solver='lbfgs', alpha=1e-2, hidden_layer_sizes=(3, 4), random_state=42)
classifier.fit(train_data_array, train_labels)