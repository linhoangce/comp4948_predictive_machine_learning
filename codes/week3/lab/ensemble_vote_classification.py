import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from mlxtend.classifier import EnsembleVoteClassifier
from xgboost import XGBClassifier, plot_importance
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def evaluate(model, X_test, y_test, title):
    print('\n*** '+ title)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

def main():
    PATH = r"C:\Users\linho\Desktop\CST\term4\pa\data\iris_numeric.csv"

    df = pd.read_csv(PATH)
    X = df.drop('iris_type', axis=1)
    y = df['iris_type']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ada_boost = AdaBoostClassifier()
    grad_boost = GradientBoostingClassifier()
    xgb = XGBClassifier()
    knn = KNeighborsClassifier()

    classifiers = [ada_boost, grad_boost, xgb, knn]

    for clf in classifiers:
        clf.fit(X_train_scaled, y_train)
        evaluate(clf, X_test_scaled, y_test, clf.__class__.__name__)

    ensemble_clf = EnsembleVoteClassifier(clfs=classifiers, voting='hard')
    ensemble_clf.fit(X_train_scaled, y_train)
    evaluate(ensemble_clf, X_test_scaled, y_test,'Ensemble Vote Classifier')


if __name__ == '__main__':
    main()