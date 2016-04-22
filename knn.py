from sklearn.neighbors import KNeighborsClassifier
import csv
import numpy as np

train = []
labels = []
test = []

reader = csv.reader(open("train.csv", "r"), delimiter=",")
next(reader)
for row in reader:
    labels.append(int(row[0]))
    row = row[1:]
    train.append(np.array(np.int64(row)))

reader = csv.reader(open("test.csv", "r"), delimiter=",")
next(reader)
for row in reader:
    test.append(np.array(np.int64(row)))

knn = KNeighborsClassifier(n_neighbors=3)
knn = knn.fit(train, labels)
out = knn.predict(test)
np.savetxt('out.csv', out, delimiter=',', fmt='%s')
