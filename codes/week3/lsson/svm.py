import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

def draw_plot(X, y, y_train, X_train_scaled, svm):
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y)
    # create a grid in scaled feature space
    x_min = X_train_scaled[:, 0].min() - 1
    x_max = X_train_scaled[:, 0].max() + 1
    y_min = X_train_scaled[:, 1].min() - 1
    y_max = X_train_scaled[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 500),
        np.linspace(y_min, y_max, 500)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = svm.predict(grid).reshape(xx.shape)

    # decision function for boundary line
    df = svm.decision_function(grid).reshape(xx.shape)

    plt.figure(figsize=(9, 6))
    plt.rcParams.update({'font.size': 18})
    # background class regions
    plt.contourf(xx, yy, Z, alpha=0.25)
    # decision boundary
    plt.contour(xx, yy, df, levels=[-1, 0, 1], linewidths=1.5)

    # training points
    plt.scatter(
        X_train_scaled[:, 0], X_train_scaled[:, 1],
        c=y_train, edgecolor='k', linewidth=0.3, alpha=0.9,
        label='Training points'
    )

    # support vector (circled)
    sv = svm.support_vectors_
    plt.scatter(
        sv[:, 0], sv[:, 1],
        s=140, facecolors='none', edgecolor='k', linewidth=1.8,
        label='Support vectors'
    )
    plt.xlabel('enagement_score')
    plt.ylabel('visit_frequency')
    plt.title('Custom Conversion')
    plt.legend()
    plt.tight_layout()
    plt.show()

def create_data():
    np.random.seed(42)
    X,y = make_moons(n_samples=400, noise=0.25, random_state=42)
    X = pd.DataFrame(X, columns=['engagement_score', 'visit_frequency'])
    y = pd.Series(y, name='converted_customer')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, X, y

def main():
    X_train_scaled, X_test_scaled, y_train, y_test, X, y = create_data()

    svm = SVC(kernel='linear', C=1)
    svm.fit(X_train_scaled, y_train)

    draw_plot(X, y, y_train, X_train_scaled, svm)
    y_pred = svm.predict(X_test_scaled)
    print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}')
    print(f'Classification Report\n{classification_report(y_test, y_pred)}')


if __name__ == '__main__':
    main()