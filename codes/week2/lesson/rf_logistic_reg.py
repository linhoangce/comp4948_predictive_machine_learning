from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd

iris = datasets.load_iris()

X = pd.DataFrame(iris.data,
                 columns=['sepal length', 'sepal width',
                          'petal length', 'petal width'])
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,
)

sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

model = LogisticRegression(fit_intercept=True, solver='lbfgs')
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print('LogisticRegression')
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print(f'Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}')
print(f'Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}')
print(f'F1: {f1_score(y_test, y_pred, average='weighted'):.4f}')

rf = RandomForestClassifier(n_estimators=200, max_features=3)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print('\nRandomForestClassifier')
print(f'Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}')
print(f'Precision: {precision_score(y_test, y_pred_rf, average='weighted'):.4f}')
print(f'Recall: {recall_score(y_test, y_pred_rf, average='weighted'):.4f}')
print(f'F1: {f1_score(y_test, y_pred_rf, average='weighted'):.4f}')