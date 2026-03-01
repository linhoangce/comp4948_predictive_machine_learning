import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def evaluate_model(model, X_test, y_test, title):
    print('\n***' + title + '***')
    predictions = model.predict(X_test)
    f1 = f1_score(y_test, predictions, average='weighted')
    acc = accuracy_score(y_test, predictions)
    print(f'F1: {f1}')
    print(f'Accuracy: {acc}')

def main():
    PATH = r"C:\Users\linho\Desktop\CST\term4\pa\data\housing_classification.csv"

    df = pd.read_csv(PATH)
    print(df.head())

    X = df.drop('price', axis=1)
    y = df['price']

    knn = KNeighborsClassifier()
    svc = SVC()
    ridge = RidgeClassifier()
    dt = DecisionTreeClassifier()

    clf_arr = [knn, svc, ridge, dt]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for clf in clf_arr:
        model_type = clf.__class__.__name__

        model = clf.fit(X_train_scaled, y_train)
        evaluate_model(model, X_test_scaled, y_test, model_type)

        bagging_clf = BaggingClassifier(clf, max_samples=0.4,
                                        max_features=6, n_estimators=1000,
                                        bootstrap=True)
        bagged_model = bagging_clf.fit(X_train_scaled, y_train)
        evaluate_model(bagged_model, X_test_scaled, y_test, f'Bagged: {model_type}')

if __name__ == '__main__':
    main()