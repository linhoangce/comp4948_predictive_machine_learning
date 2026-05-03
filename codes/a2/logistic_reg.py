import numpy as np
import pandas as pd
import pickle
from sklearn.feature_selection import RFE, RFECV, f_regression, SelectKBest, chi2
from sklearn.impute import KNNImputer
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE

DECORATOR = "=" * 120

# =============================================================================
# OPTIMAL FEATURE SELECTION METHODS
# =============================================================================

def select_features_rfe(X, y, n_features):
    rfe = RFE(LogisticRegression(solver='liblinear', max_iter=2000, random_state=42),
              n_features_to_select=n_features)
    rfe.fit(X, y)
    return list(X.columns[rfe.support_])


def select_features_ffs(X, y, n_features):
    f_stats, _ = f_regression(X, y)
    df_ffs = pd.DataFrame({'features': X.columns, 'f_stats': f_stats})
    df_ffs.sort_values(by=['f_stats'], ascending=False, inplace=True)
    top_features = df_ffs['features'].head(n_features)
    return list(top_features)


def select_features_chi2(X, y, n_features):
    # Scale X values to avoid negative values as required by chi2
    X_scaled = X - X.min() + 1e-6
    selector = SelectKBest(score_func=chi2, k=n_features)
    selector.fit(X_scaled, y)
    return list(X.columns[selector.get_support()])


def build_features_pool(X, y, scaler, k):
    """
    Build a feature pool from top-ks of RFE, FFS, and Chi2
    """
    print('\n' + DECORATOR)
    print("BUILDING FEATURE POOL FROM RFE, FFS, AND CHI2")
    print(DECORATOR + '\n')

    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    top_k = k
    pool = set()

    rfe_features = select_features_rfe(X_scaled, y, top_k)
    print(f'RFE top {top_k} features:\n{rfe_features}\n')
    pool.update(rfe_features)

    ffs_features = select_features_ffs(X_scaled, y, top_k)
    print(f'FFS top {top_k} features:\n{ffs_features}\n')
    pool.update(ffs_features)

    chi2_features = select_features_chi2(X_scaled, y, top_k)
    print(f'Chi2 top {top_k} features:\n{chi2_features}\n')
    pool.update(chi2_features)

    pool = list(pool)
    print(f'\nTotal unique features in pool: {len(pool)}\n')
    return pool


def evaluate_feature_subset_cv(X, y, features, scaler):
    k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    f1s, precisions, recalls, accuracies = [], [], [], []

    for train_idx, test_idx in k_fold.split(X, y):
        X_train_fold = X.iloc[train_idx][features].copy()
        X_test_fold = X.iloc[test_idx][features].copy()
        y_train_fold = y.iloc[train_idx].copy()
        y_test_fold = y.iloc[test_idx].copy()

        # create new instance of scaler for cross-fold validation
        scaler_fold = type(scaler)()

        X_train_scaled = scaler_fold.fit_transform(X_train_fold)
        X_test_scaled = scaler_fold.transform(X_test_fold)

        X_smote, y_smote = SMOTE(random_state=42).fit_resample(X_train_scaled, y_train_fold)

        model = LogisticRegression(
            solver='liblinear',
            max_iter=2000,
            random_state=42,
            class_weight={0: 1.0, 1: 3.0},
        )
        model.fit(X_smote, y_smote)

        y_pred = model.predict(X_test_scaled)

        f1s.append(f1_score(y_test_fold, y_pred))
        precisions.append(precision_score(y_test_fold, y_pred))
        recalls.append(recall_score(y_test_fold, y_pred))
        accuracies.append(accuracy_score(y_test_fold, y_pred))

    return {
        "F1": np.mean(f1s),
        "Precision": np.mean(precisions),
        "Recall": np.mean(recalls),
        "Accuracy": np.mean(accuracies),
    }


def build_feature_pool_experiments(X_train, y_train, scaler, top_k=20):
    """
    Use RFE, FFS, Chi2 as GUIDES (not in CV) to build a pool of promising features.
    This runs ONCE on full training data - these are just recommendations.
    """
    print("\n" + DECORATOR)
    print("BUILDING FEATURE POOL (GUIDES ONLY - NO CV)")
    print(DECORATOR + "\n")

    # Scale once for all selectors
    X_scaled = scaler.fit_transform(X_train)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_train.columns)

    pool = {}

    # Guide 1: RFE top features
    print(f"Running RFE (top {top_k})...")
    rfe_features = select_features_rfe(X_scaled_df, y_train, top_k)
    pool['RFE'] = rfe_features
    print(f"  RFE suggests: {rfe_features}\n")

    # Guide 2: FFS top features
    print(f"Running FFS (top {top_k})...")
    ffs_features = select_features_ffs(X_scaled_df, y_train, top_k)
    pool['FFS'] = ffs_features
    print(f"  FFS suggests: {ffs_features}\n")

    # Guide 3: Chi2 top features
    print(f"Running Chi2 (top {top_k})...")
    chi2_features = select_features_chi2(X_scaled_df, y_train, top_k)
    pool['Chi2'] = chi2_features
    print(f"  Chi2 suggests: {chi2_features}\n")

    # Combined pool (union of all suggestions)
    all_features = set(rfe_features) | set(ffs_features) | set(chi2_features)

    # Feature frequency analysis
    feature_counts = {}
    for feature in all_features:
        count = sum([
            1 if feature in rfe_features else 0,
            1 if feature in ffs_features else 0,
            1 if feature in chi2_features else 0
        ])
        feature_counts[feature] = count

    # Sort by frequency (how many methods selected it)
    sorted_pool = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)

    print("Feature Pool Analysis:")
    print("-" * 80)
    print(f"{'Feature':<40s} {'Selected by (count)'}")
    print("-" * 80)
    for feature, count in sorted_pool:
        methods = []
        if feature in rfe_features:
            methods.append('RFE')
        if feature in ffs_features:
            methods.append('FFS')
        if feature in chi2_features:
            methods.append('Chi2')
        print(f"{feature:<40s} {', '.join(methods)} ({count}/3)")

    print(f"\nTotal features in pool: {len(all_features)}")
    print(f"Features selected by all 3 methods: {sum(1 for c in feature_counts.values() if c == 3)}")
    print(f"Features selected by 2+ methods: {sum(1 for c in feature_counts.values() if c >= 2)}")

    return pool, list(all_features), sorted_pool


def experiment_with_combinations(X_train, y_train, feature_pool, sorted_pool, scaler):
    print("\n" + DECORATOR)
    print("EXPERIMENTING WITH FEATURE COMBINATIONS")
    print(DECORATOR + "\n")

    experiments = []

    # Experiment 1: Features selected by all 3 methods (highest confidence)
    features_common = [f for f, count in sorted_pool if count == 3]
    if len(features_common) >= 3:
        print(f'Experiment 1: Most common features (selected by all 3 selection algorithm), '
              f'n={len(features_common)}')
        score = evaluate_feature_subset_cv(X_train, y_train, features_common, scaler)
        experiments.append({
            'name': 'Consensus (3/3)',
            'features': features_common,
            'score': score['F1']
        })
        print(f'   Class 1 F1: {score["F1"]:.4f}\n')

    # Experiment 2: Features selected by 2+ algorithms
    high_confidence = [f for f, count in sorted_pool if count >= 2]
    if len(high_confidence) >= 5:
        print(f'Experiment 2: High Confidence features (2+ algorithms), '
              f'n={len(high_confidence)}')
        score = evaluate_feature_subset_cv(X_train, y_train, high_confidence, scaler)
        experiments.append({
            'name': 'High Confidence (2+/3)',
            'features': high_confidence,
            'score': score['F1']
        })
        print(f'   Class 1 F1: {score["F1"]:.4f}\n')

    # Experiment 3: top 10 from pool
    top_10 = [f for f, count in sorted_pool[:25]]
    print(f'Experiment 3: Top 10 Features from Feature Pool')
    score = evaluate_feature_subset_cv(X_train, y_train, top_10, scaler)
    experiments.append({
        'name': 'Top 10',
        'features': top_10,
        'score': score['F1']
    })
    print(f"  Macro F1: {score['F1']:.4f}\n")

    # Experiment 4: Top 15
    top_15 = [f for f, count in sorted_pool[:35]]
    print(f'Experiment 4: Top 15 features from Feature Pool')
    score = evaluate_feature_subset_cv(X_train, y_train, top_15, scaler)
    experiments.append({
        'name': 'Top 15',
        'features': top_15,
        'score': score['F1']
    })
    print(f"  Macro F1: {score['F1']:.4f}\n")

    # Experiment 5: Top 20
    top_20 = [f for f, count in sorted_pool[:50]]
    print(f"Experiment 5: Top 20 by frequency")
    score = evaluate_feature_subset_cv(X_train, y_train, top_20, scaler)
    experiments.append({
        'name': 'Top 20',
        'features': top_20,
        'score': score['F1']
    })
    print(f"  Macro F1: {score['F1']:.4f}\n")

    # Experiment 6: RFE only
    print(f'Experiment 6: RFE features only (n={len(feature_pool["RFE"])})')
    score = evaluate_feature_subset_cv(X_train, y_train, feature_pool['RFE'], scaler)
    experiments.append({
        'name': 'RFE only',
        'features': feature_pool['RFE'],
        'score': score['F1']
    })
    print(f'   Class 1 F1: {score["F1"]:.4f}\n')

    # Experiment 7: FFS only
    print(f'Experiment 7: FFS features only (n={len(feature_pool["FFS"])})')
    score = evaluate_feature_subset_cv(X_train, y_train, feature_pool['FFS'], scaler)
    experiments.append({
        'name': 'FFS only',
        'features': feature_pool['FFS'],
        'score': score['F1']
    })
    print(f'   Class 1 F1: {score["F1"]:.4f}\n')

    # Experiment 8: Chi2 suggestions only
    print(f"Experiment 8: Chi2 features only (n={len(feature_pool['Chi2'])})")
    score = evaluate_feature_subset_cv(X_train, y_train, feature_pool['Chi2'], scaler)
    experiments.append({
        'name': 'Chi2 Only',
        'features': feature_pool['Chi2'],
        'score': score['F1']
    })
    print(f"  Macro F1: {score['F1']:.4f}\n")

    # Experiment 9: Intersection of RFE and FFS
    rfe_ffs_intersection = list(set(feature_pool['RFE']) & set(feature_pool['FFS']))
    if len(rfe_ffs_intersection) >= 5:
        print(f'Experiment 9: RFE intersects FFS (n={len(rfe_ffs_intersection)})')
        score = evaluate_feature_subset_cv(X_train, y_train, rfe_ffs_intersection, scaler)
        experiments.append({
            'name': 'RFE intersects FFS',
            'features': rfe_ffs_intersection,
            'score': score['F1']
        })
        print(f'   Class 1 F1: {score["F1"]:.4f}\n')

    # Experiment 10: Union of top 20 from each method
    rfe_top20 = feature_pool['RFE'][:50]
    ffs_top20 = feature_pool['FFS'][:50]
    chi2_top20 = feature_pool['Chi2'][:50]
    union_top20 = list(set(rfe_top20) | set(ffs_top20) | set(chi2_top20))

    print(f'Experiment 10: Union of top 20 from each algorithm (n={len(chi2_top20)})')
    score = evaluate_feature_subset_cv(X_train, y_train, union_top20, scaler)
    experiments.append({
        'name': 'Union Top 20',
        'features': union_top20,
        'score': score['F1']
    })
    print(f'   Class 1 F1: {score["F1"]:.4f}\n')

    experiments.sort(key=lambda x: x['score'], reverse=True)

    print(DECORATOR)
    print("EXPERIMENT RESULTS SUMMARY")
    print(DECORATOR + "\n")

    for i, exp in enumerate(experiments, 1):
        print(f' {i}. {exp["name"]:.25s} | F1={exp["score"]:.4f} | n={len(exp["features"])}')

    print(f'\n\nBEST COMBINATION: {experiments[0]["features"]} (F1={experiments[0]["score"]:.4f})')

    return experiments[0]['features'], experiments[0]['score'], experiments



def train_final_model(X_train, y_train, features, scaler):
    X_selected = X_train[features]
    X_scaled = scaler.fit_transform(X_selected)
    X_smote, y_smote = SMOTE(random_state=42).fit_resample(X_scaled, y_train)

    model = LogisticRegression(
        solver='liblinear',
        max_iter=5000,
        random_state=42,
        class_weight={0: 1.0, 1: 3.0},
    )
    model.fit(X_smote, y_smote)
    return model, scaler


def evaluate_final_model(model, scaler, X_test, y_test, features):
    X_selected = X_test[features]
    X_scaled = scaler.transform(X_selected)

    y_pred = model.predict(X_scaled)

    print(f'\nClassification Report:\n{classification_report(y_test, y_pred)}')
    print(f'\nMacro F1 (TRUE performance): {f1_score(y_test, y_pred, average="macro"):.4f}')

    return {
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred, pos_label=1),
        "Macro_F1": f1_score(y_test, y_pred, average='macro'),
        "Accuracy": accuracy_score(y_test, y_pred)
    }


def main():
    df = pd.read_csv("data/bankruptcy_asgn2.csv")
    X = df.drop(columns=["Bankrupt?"])
    y = df["Bankrupt?"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()

    print("\n" + DECORATOR)
    print(DECORATOR)
    print("PERFORMING FEATURE SELECTION & EXPERIMENT WITH CV EVALUATION")
    print(DECORATOR)
    print(DECORATOR + "\n")

    pool, all_features, sorted_pool = build_feature_pool_experiments(X_train, y_train, scaler, top_k=60)

    best_features, cv_score, all_experiments = experiment_with_combinations(
        X_train, y_train, pool, sorted_pool, scaler
    )

    print("\n" + DECORATOR)
    print(DECORATOR)
    print("RUNNING ITERATIONS OF EVALUATION OF SELECTED FEATURE COMBINATIONS ")
    print(DECORATOR)
    print(DECORATOR + "\n")


    # ----------------------------------------------
    # STORE METRICS OVER 50 RUNS
    # ----------------------------------------------
    results = {
        "Precision": [],
        "Recall": [],
        "F1": [],
        "Macro_F1": [],
        "Accuracy": []
    }

    for run in range(50):
        print(f"\n\n========== RUN {run + 1} / 50 ==========\n")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model, scaler = train_final_model(X_train, y_train, best_features, StandardScaler())
        metrics = evaluate_final_model(model, scaler, X_test, y_test, best_features)

        print(metrics["F1"])

if __name__ == '__main__':
    main()