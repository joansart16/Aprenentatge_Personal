import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


digits = datasets.load_digits()


#Preprocessament
digits2 = []
for d in digits.images:
    digits2.append(np.reshape(d, 64))

X_train, X_test, y_train, y_test = train_test_split(digits2, digits.target, test_size = 0.3, random_state = 5)

clf = SGDClassifier(loss = "log", eta0 = 0.1, max_iter=1000, learning_rate="constant", random_state = 5)

clf.fit(X_train, y_train)
prediction = clf.predict(X_test)

#Avaluació
cf_matrix = confusion_matrix(y_test, prediction)
print(cf_matrix)
accuracy = accuracy_score(y_test, prediction)
print("Accuracy: ", accuracy)

numeros = ["0","1","2","3","4","5","6","7","8","9",]
print(classification_report(y_test, prediction, target_names = numeros))
