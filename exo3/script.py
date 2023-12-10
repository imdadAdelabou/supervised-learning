import numpy as np
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.metrics import accuracy_score


X_test = np.load('datasets/X_test.npy')
y_test = np.load('datasets/y_test.npy')
X = np.load('datasets/X_train.npy')
y = np.load('datasets/y_train.npy')

print(np.shape(X)) #(500, 50)
print(np.shape(y)) #(500,)
knn = KNeighborsClassifier(n_neighbors=3) 
knn.fit(X=X, y=y)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: {} with k= {}".format(accuracy, 3))
