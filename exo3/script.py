import numpy as np
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score


#Load the data
X_test = np.load('datasets/X_test.npy')
y_test = np.load('datasets/y_test.npy')
X = np.load('datasets/X_train.npy')
y = np.load('datasets/y_train.npy')


#Generate a list of K values
k_values = [i for i in range(1, 60)]
scores = []

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(np.shape(X)) #(500, 50)
print(np.shape(y)) #(500,)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X, y, cv=5)
    scores.append(np.mean(score))

best_index = np.argmax(scores)
best_k = k_values[best_index]
print(best_k)
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X=X, y=y)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: {} with k= {}".format(accuracy, best_k))
