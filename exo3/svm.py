import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# Load data
X_test = np.load('datasets/X_test.npy')
y_test = np.load('datasets/y_test.npy')
X = np.load('datasets/X_train.npy')
y = np.load('datasets/y_train.npy')

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Print data dimensions
print(np.shape(X))  # (500, 50)
print(np.shape(y))  # (500,)

# SVM Parameters
C_values = [0.1, 1, 10, 100]  # Different values for the regularization parameter C
kernel = 'linear'  # Linear kernel for simplicity

# Cross-validation to find the best C
scores = []

for C in C_values:
    svm_classifier = SVC(C=C, kernel=kernel)
    score = cross_val_score(svm_classifier, X_scaled, y, cv=5)
    scores.append(np.mean(score))

best_index = np.argmax(scores)
best_C = C_values[best_index]
print("Best C:", best_C)

# Train the final model with the best C
svm_classifier = SVC(C=best_C, kernel=kernel)
svm_classifier.fit(X_scaled, y)

# Test the model
X_test_scaled = scaler.transform(X_test)
y_pred = svm_classifier.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy with SVM: {} (C={})".format(accuracy, best_C))
