import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              precision_score, recall_score)
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class MLEvaluator:
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'MLP Neural Net': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
        }
        self.results = {}

    def _prepare_Xy(self, df, target_col_idx=-1):
        arr = df.values if hasattr(df, 'values') else df
        X = arr[:, :target_col_idx]
        y = (arr[:, target_col_idx] > 0.5).astype(int)
        return X, y

    def train_test_utility(self, real_df, synthetic_df, target_col_idx=-1):
        """TSTR: Train on Synthetic, Test on Real"""
        logger.info("Running TSTR evaluation...")
        X_real, y_real = self._prepare_Xy(real_df, target_col_idx)
        X_syn, y_syn = self._prepare_Xy(synthetic_df, target_col_idx)

        scaler = StandardScaler()
        X_syn_s = scaler.fit_transform(X_syn)
        X_real_s = scaler.transform(X_real)

        tstr_results = {}
        trtr_results = {}

        for name, model in self.models.items():
            # TSTR
            try:
                m_tstr = type(model)(**model.get_params())
                m_tstr.fit(X_syn_s, y_syn)
                y_pred = m_tstr.predict(X_real_s)
                y_prob = m_tstr.predict_proba(X_real_s)[:, 1] if hasattr(m_tstr, 'predict_proba') else y_pred
                tstr_results[name] = {
                    'accuracy': float(accuracy_score(y_real, y_pred)),
                    'f1': float(f1_score(y_real, y_pred, zero_division=0)),
                    'auc': float(roc_auc_score(y_real, y_prob)),
                    'precision': float(precision_score(y_real, y_pred, zero_division=0)),
                    'recall': float(recall_score(y_real, y_pred, zero_division=0))
                }
            except Exception as e:
                logger.warning(f"TSTR {name} failed: {e}")
                tstr_results[name] = {'accuracy': 0, 'f1': 0, 'auc': 0.5, 'precision': 0, 'recall': 0}

            # TRTR (baseline)
            try:
                m_trtr = type(model)(**model.get_params())
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                scores = cross_val_score(m_trtr, X_real_s, y_real, cv=cv, scoring='accuracy')
                trtr_results[name] = {
                    'accuracy': float(np.mean(scores)),
                    'std': float(np.std(scores))
                }
            except Exception as e:
                logger.warning(f"TRTR {name} failed: {e}")
                trtr_results[name] = {'accuracy': 0.5, 'std': 0}

        utility_gap = {}
        for name in tstr_results:
            gap = trtr_results[name]['accuracy'] - tstr_results[name]['accuracy']
            utility_gap[name] = float(gap)

        overall_tstr_acc = np.mean([v['accuracy'] for v in tstr_results.values()])
        overall_trtr_acc = np.mean([v['accuracy'] for v in trtr_results.values()])

        self.results = {
            'tstr': tstr_results,
            'trtr': trtr_results,
            'utility_gap': utility_gap,
            'overall_tstr_accuracy': float(overall_tstr_acc),
            'overall_trtr_accuracy': float(overall_trtr_acc),
            'utility_preservation': float(max(0, 1 - np.mean(list(utility_gap.values()))) * 100)
        }
        return self.results

    def feature_importance_analysis(self, real_df, target_col_idx=-1):
        """Compute feature importances on real data"""
        X, y = self._prepare_Xy(real_df, target_col_idx)
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        rf = RandomForestClassifier(n_estimators=200, random_state=42)
        rf.fit(X_s, y)
        importances = rf.feature_importances_
        cols = list(real_df.columns)[:-1] if hasattr(real_df, 'columns') else [f'f{i}' for i in range(X.shape[1])]
        return dict(zip(cols, importances.tolist()))
