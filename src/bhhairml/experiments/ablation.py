from __future__ import annotations
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

def classification_ablation(feature_sets, y_train, y_test, train_idx, test_idx, seed=42):
    rows = []
    for name, X in feature_sets.items():
        model = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
        model.fit(X[train_idx], y_train)
        rows.append({"feature_set": name, "metric": "macro F1",
                     "score": f1_score(y_test, model.predict(X[test_idx]), average="macro")})
    return pd.DataFrame(rows)
