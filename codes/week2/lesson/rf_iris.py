from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import datasets
import pandas as pd

iris = datasets.load_iris()

y = iris.target

X = pd.DataFrame(iris.data, columns=['sepal length', 'sepal width',
                                     'petal length', 'petal width'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=True
)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
# print(rf.predict([[3, 5, 4, 2]]))

importances = rf.feature_importances_

feature_imp = pd.DataFrame([{'feature': X.columns[i],
                             'importance': importances[i]}
                            for i in range(len(importances))])
feature_imp = feature_imp.sort_values(by=['importance'], ascending=False)
print(feature_imp)

features = list(X.columns)
top_indices = [features.index('petal width'), features.index('petal length')]

rf_top_features = RandomForestClassifier(n_estimators=100)
top_train = X_train.iloc[:, top_indices]
top_test = X_test.iloc[:, top_indices]

rf_top_features.fit(top_train, y_train)
y_pred_top = rf_top_features.predict(top_test)

print(f'Accuracy top features: {accuracy_score(y_test, y_pred_top):.4f}')