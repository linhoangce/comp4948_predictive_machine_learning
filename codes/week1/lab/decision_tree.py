import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.tree import plot_tree

path = r"C:\Users\linho\Desktop\CST\term4\pa\data\bill_authentication.csv"
df = pd.read_csv(path)

X = df.drop(columns=['Class'], axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

classifier = DecisionTreeClassifier(max_depth=2)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print('\nModel Evaluation')
cm = confusion_matrix(y_test, y_pred)
print(cm)

print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Precision: {precision_score(y_test, y_pred)}')
print(f'Recall: {recall_score(y_test, y_pred)}')

fig, ax = plt.subplots(figsize=(20, 10))

plot_tree(classifier.fit(X_train, y_train),
          max_depth=4, fontsize=4)
a = plot_tree(classifier,
              feature_names=['Variance', 'Skewness', 'Kurtosis', 'Entropy'],
              filled=True,
              rounded=True,
              fontsize=14)
# plt.show()

# reconstruct the tree manually and calculate fraudulent and non-fraud counts
# with training data
def manually_classify(X, y):
    predictions = []
    zeros, ones = 0, 0

    for i in range(len(X)):
        if(X.iloc[i]['Variance'] <= 0.274): # blue
            if(X.iloc[i]['Skewness'] <= 7.565):
                ones += 1
            else:
                zeros += 1
        else:
            if X.iloc[i]['Kurtosis'] <= -4.394:
                ones += 1
            else:
                zeros += 1

    print(f'Zeros: {zeros}')
    print(f'One:s {ones}')

manually_classify(X_train, y_train)