import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import make_classification
from sklearn.svm import SVC


def make_data(seed=42, n_samples=600):
    np.random.seed(seed)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_clusters_per_class=2,
        class_sep=1.2,
        flip_y=0.05,
        random_state=seed
    )
    X = pd.DataFrame(X,
                     columns=['avg_transaction_value',
                              'purchase_frequency',
                              'customer_tenure',
                              'engagement_score'])

    y = pd.Series(y, name='high_value_customer')
    return X, y

def split_and_scale(X, y, features, test_size=0.2, random_state=42):
    X3 = X[features].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X3, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_and_evaluate_svm(X_train_scaled, X_test_scaled, y_train, y_test,
                           kernel='rbf', C=0.1, gamma='scale'):
    svm = SVC(kernel=kernel, C=C, gamma=gamma)
    svm.fit(X_train_scaled, y_train)
    y_pred = svm.predict(X_test_scaled)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return svm

def approx_boundary_points_3d(svm, X_train_scaled, grid_n=100,
                              pad=0.75, tol=0.06):
    x_min = X_train_scaled[:, 0].min() - pad
    x_max = X_train_scaled[:, 0].max() + pad
    y_min = X_train_scaled[:, 1].min() - pad
    y_max = X_train_scaled[:, 1].max() + pad
    z_min = X_train_scaled[:, 2].min() - pad
    z_max = X_train_scaled[:, 2].max() + pad

    xs = np.linspace(x_min, x_max, grid_n)
    ys = np.linspace(y_min, y_max, grid_n)
    zs = np.linspace(z_min, z_max, grid_n)

    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing='ij')
    grid_points = np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()]

    df = svm.decision_function(grid_points)
    boundary_mask = np.abs(df) < tol
    boundary_points = grid_points[boundary_mask]
    return boundary_points

def plot_3d_svm(X_train_scaled, y_train, svm, features, boundary_points=None):
    y_train_np = y_train.to_numpy()
    mask0 = y_train_np == 0
    mask1 = y_train_np == 1
    fig = go.Figure()

    # Class 0
    fig.add_trace(go.Scatter3d(
        x=X_train_scaled[mask0, 0],
        y=X_train_scaled[mask0, 1],
        z=X_train_scaled[mask0, 2],
        mode='markers',
        name='Class 0',
        marker=dict(size=3, opacity=0.75)
    ))
    # Class 1
    fig.add_trace(go.Scatter3d(
        x=X_train_scaled[mask1, 0],
        y=X_train_scaled[mask1, 1],
        z=X_train_scaled[mask1, 2],
        mode='markers',
        name='Class 1',
        marker=dict(size=3, opacity=0.75)
    ))

    # support vector
    sv = svm.support_vectors_
    fig.add_trace(go.Scatter3d(
        x=sv[:, 0], y=sv[:, 1], z=sv[:, 2],
        mode='markers', name='Support Vector',
        marker=dict(size=7, symbol='circle-open', opacity=1.0, line=dict(width=3))
    ))
    # boundary point cloude
    if boundary_points is not None and boundary_points.shape[0] > 0:
        fig.add_trace(go.Scatter3d(
            x=boundary_points[:, 0],
            y=boundary_points[:, 1],
            z=boundary_points[:, 2],
            mode='markers',
            name='Decision Boundary (approx)',
            marker=dict(size=2, opacity=0.12)
        ))
    else:
        print('No boundary points found. Try increasing tol or grid_n')

    fig.update_layout(
        title='Rotatable 3D SVM (RBF)',
        scene=dict(
            xaxis_title=f'{features[0]} (scaled)',
            yaxis_title=f'{features[1]} (scaled)',
            zaxis_title=f'{features[2]} (scaled)'
        ),
        legend=dict(itemsizing='constant')
    )
    fig.show()


def main():
    FEATURES = ['avg_transaction_value', 'purchase_frequency', 'customer_tenure']

    X, y = make_data()
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = \
        split_and_scale(X, y, FEATURES)
    svm = train_and_evaluate_svm(X_train_scaled, X_test_scaled, y_train, y_test)
    boundary_points = approx_boundary_points_3d(
        svm, X_train_scaled, grid_n=100, pad=0.75, tol=0.08
    )
    plot_3d_svm(X_train_scaled, y_train, svm, FEATURES, boundary_points)


if __name__ == '__main__':
    main()