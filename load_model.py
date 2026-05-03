"""
load_model.py  —  version-independent model loader
Drop this file next to model_portable.json and import it in app.py.

Usage in app.py:
    from load_model import load_portable_model
    model = load_portable_model('model_portable.json')
    prediction = model.predict(X)          # 0=control, 1=dementia
    probability = model.predict_proba(X)   # shape (n, 2)
"""

import json
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree._classes import DecisionTreeRegressor
from sklearn.tree._tree import Tree
import ctypes


def _rebuild_tree(node_data, n_features):
    """Reconstruct a single decision tree from raw node arrays."""
    tree = DecisionTreeRegressor(max_depth=None)
    # Fit on a tiny dummy so internal structure is allocated
    tree.fit([[0] * n_features], [[0]])

    t = tree.tree_
    n_nodes = len(node_data['children_left'])

    # Resize internal arrays
    t.__setstate__({
        'max_depth':       int(max(node_data['children_left'].__class__.__mro__[0].__name__ and
                               np.array(node_data['children_left']).max() >= 0 and 1) or 1),
        'node_count':      n_nodes,
        'nodes':           np.zeros(n_nodes, dtype=np.dtype({
            'names':   ['left_child','right_child','feature','threshold',
                        'impurity','n_node_samples','weighted_n_node_samples','missing_go_to_left'],
            'formats': ['<i8','<i8','<i8','<f8','<f8','<i8','<f8','u1'],
            'offsets': [0,8,16,24,32,40,48,56],
            'itemsize': 64
        })),
        'values':          np.array(node_data['value']),
        'n_features':      n_features,
        'n_outputs':       1,
        'n_classes':       np.array([1], dtype=np.intp),
    })
    return tree


def load_portable_model(json_path: str):
    """
    Load the Alzheimer's detection model from a JSON file.
    Works regardless of numpy / sklearn version.
    Returns a simple object with .predict() and .predict_proba() methods.
    """
    with open(json_path) as f:
        data = json.load(f)

    scaler_mean  = np.array(data['scaler_mean'])
    scaler_scale = np.array(data['scaler_scale'])
    params       = data['clf_params']
    init_prior   = np.array(data['init_prediction'])   # class prior from DummyClassifier
    stages       = data['estimators']
    learning_rate = params['learning_rate']

    class PortableModel:
        def _scale(self, X):
            X = np.array(X, dtype=float)
            return (X - scaler_mean) / scaler_scale

        def _traverse(self, tree_data, x):
            cl = tree_data['children_left']
            cr = tree_data['children_right']
            feat = tree_data['feature']
            thr  = tree_data['threshold']
            val  = tree_data['value']
            node = 0
            while cl[node] != -1:
                if x[feat[node]] <= thr[node]:
                    node = cl[node]
                else:
                    node = cr[node]
            return val[node][0][0]

        def decision_function(self, X):
            X = self._scale(X)
            # Start from log-odds of prior
            log_odds_prior = np.log(init_prior[1] / max(init_prior[0], 1e-10))
            scores = np.full(len(X), log_odds_prior)
            for stage in stages:
                for tree_data in stage:
                    for i, x in enumerate(X):
                        scores[i] += learning_rate * self._traverse(tree_data, x)
            return scores

        def predict_proba(self, X):
            scores = self.decision_function(X)
            prob_pos = 1 / (1 + np.exp(-scores))
            return np.column_stack([1 - prob_pos, prob_pos])

        def predict(self, X):
            return (self.decision_function(X) > 0).astype(int)

    return PortableModel()


if __name__ == '__main__':
    import os
    base = os.path.dirname(__file__)
    m = load_portable_model(os.path.join(base, 'model_portable.json'))
    # Smoke test with zeros
    X_test = np.zeros((2, 18))
    print("predict:       ", m.predict(X_test))
    print("predict_proba: ", m.predict_proba(X_test).round(3))
    print("load_model.py OK")
