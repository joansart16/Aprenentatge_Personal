from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=2,
                           random_state=5)

# Tractament de les dades: Separació i estandaritzat
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Entrenament i predicció
clf = SGDClassifier(loss="perceptron", eta0=1, max_iter=1000, learning_rate="constant", random_state=5)
clf.fit(X_train_scaled, y_train)
prediction = clf.predict(X_test_scaled)

# Avaluació
cf_matrix = confusion_matrix(y_test, prediction)
print(cf_matrix)
accuracy = accuracy_score(y_test, prediction)
print("Accuracy: ", accuracy)


#F1 2*(Precisio*Sensibilitat)/(Precisio + Sensibilitat)
#Presició = TP/(TP+FP)
#Sensibilitat = TPR = TP/(FN+TP)

precisio = cf_matrix[0][0] / (cf_matrix[0][0] + cf_matrix[1][0])
sensibilitat = cf_matrix[0][0]/(cf_matrix[0][1]+cf_matrix[0][0])
F1 = 2*(precisio*sensibilitat)/(precisio + sensibilitat)
print("F1 = ",F1)

origenY = -(clf.intercept_[0]/clf.coef_[0,1])
pendent = -(clf.coef_[0,0]/clf.coef_[0,1])
print("Punt origen: (0, ",origenY,")")
print("Pendent: ",pendent)
plt.figure(1)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.axline(xy1=(0, origenY), slope=pendent)
plt.title("Perceptron")

plt.show()


print("###################################")

clf = SGDClassifier(loss="log", eta0=1, max_iter=1000, learning_rate="constant", random_state=5)
clf.fit(X_train_scaled, y_train)
prediction = clf.predict(X_test_scaled)

# Avaluació
cf_matrix = confusion_matrix(y_test, prediction)
print(cf_matrix)
accuracy = accuracy_score(y_test, prediction)
print("Accuracy: ", accuracy)


#F1 2*(Precisio*Sensibilitat)/(Precisio + Sensibilitat)
#Presició = TP/(TP+FP)
#Sensibilitat = TPR = TP/(FN+TP)

precisio = cf_matrix[0][0] / (cf_matrix[0][0] + cf_matrix[1][0])
sensibilitat = cf_matrix[0][0]/(cf_matrix[0][1]+cf_matrix[0][0])
F1 = 2*(precisio*sensibilitat)/(precisio + sensibilitat)
print("F1 = ",F1)

origenY2 = -(clf.intercept_[0]/clf.coef_[0,1])
pendent2 = -(clf.coef_[0,0]/clf.coef_[0,1])
print("Punt origen: (0, ",origenY2,")")
print("Pendent: ",pendent2)
plt.figure(1)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.axline(xy1=(0, origenY), slope=pendent)
plt.axline(xy1=(0, origenY2), slope=pendent2)
plt.title("Regressio Logistica")

plt.show()

