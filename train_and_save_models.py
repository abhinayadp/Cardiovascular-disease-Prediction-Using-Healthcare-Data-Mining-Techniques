"""
Script to train all three models and save them as pickle files for the GUI.
"""
import pandas as pd
import numpy as np
import pickle
import math
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from collections import Counter
from dataclasses import dataclass, field

from Models.random_forest import CustomRandomForest
from Models.svm_model import CustomSVM

# Create models directory
Path("models").mkdir(exist_ok=True)

# ============================================================================
# 1. LOAD AND PREPROCESS DATA
# ============================================================================

def load_and_preprocess_data():
    """Load and preprocess the cardiovascular dataset."""
    # Load RAW data (not the already-scaled cardio_clean.csv)
    df = pd.read_csv('Dataset/cardio_train.csv', sep=';')
    
    # Convert age from days to years
    df['age'] = (df['age'] / 365.25).astype(int)
    
    # Basic data cleaning
    # Swap systolic and diastolic if reversed
    swapped = df['ap_hi'] < df['ap_lo']
    df.loc[swapped, ['ap_hi', 'ap_lo']] = df.loc[swapped, ['ap_lo', 'ap_hi']].values
    
    # Remove outliers
    df = df[(df['age'] >= 30) & (df['age'] <= 80)]
    df = df[(df['height'] >= 130) & (df['height'] <= 210)]
    df = df[(df['weight'] >= 40) & (df['weight'] <= 200)]
    df = df[(df['ap_hi'] >= 80) & (df['ap_hi'] <= 240)]
    df = df[(df['ap_lo'] >= 40) & (df['ap_lo'] <= 150)]
    
    # Remove duplicates
    df = df.drop_duplicates(ignore_index=True)
    
    # Feature Engineering
    df['bmi'] = df['weight'] / (df['height'] / 100) ** 2
    df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
    
    # Drop unnecessary columns
    cols_to_drop = ['cardio', 'id']
    X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    y = df['cardio']
    
    return X, y

# ============================================================================
# 2. CUSTOM MODEL CLASSES
# ============================================================================

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))

@dataclass
class LogisticRegressionScratch:
    lr: float = 0.001
    n_iter: int = 8000
    l2: float = 0.0
    random_state: int | None = 42
    print_every: int = 400

    w: np.ndarray = field(init=False)
    b: float = field(init=False, default=0.0)

    def fit(self, X: np.ndarray, y: np.ndarray):
        n, d = X.shape
        self.w = np.zeros(d, dtype=float)
        self.b = 0.0
        
        for i in range(self.n_iter):
            z = X @ self.w + self.b
            pred = _sigmoid(z)
            
            dw = (1/n) * (X.T @ (pred - y)) + (self.l2 / n) * self.w
            db = (1/n) * np.sum(pred - y)
            
            self.w -= self.lr * dw
            self.b -= self.lr * db
            
            if i % self.print_every == 0:
                loss = -np.mean(y * np.log(pred + 1e-15) + (1 - y) * np.log(1 - pred + 1e-15))
                print(f"Iteration {i}: Loss = {loss:.4f}")
        
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba_1 = _sigmoid(X @ self.w + self.b)
        proba_0 = 1.0 - proba_1
        return np.column_stack((proba_0, proba_1))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


class CustomKNN:
    def __init__(self, k: int = 5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = X
        self.y_train = y
        return self

    def _euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return float(np.linalg.norm(x1 - x2))

    def _predict(self, x_new: np.ndarray) -> int:
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model not fitted")
        distances = [self._euclidean_distance(x_new, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return int(most_common[0][0])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict(x) for x in X])


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)
        return self

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        unique_classes = np.unique(y)

        if len(unique_classes) == 1:
            return {'label': unique_classes[0]}

        if self.max_depth and depth == self.max_depth:
            most_common_class = self._most_common_class(y)
            return {'label': most_common_class}

        split_best = self._best_split(X, y)
        if not split_best:
            return {'label': self._most_common_class(y)}
            
        lefttree = self._build_tree(split_best['left_X'], split_best['left_y'], depth + 1)
        righttree = self._build_tree(split_best['right_X'], split_best['right_y'], depth + 1)

        return {
            'feature_index': split_best['feature_index'],
            'threshold': split_best['threshold'],
            'left': lefttree,
            'right': righttree
        }

    def _best_split(self, X, y):
        best_gain_info = -float('inf')
        best_split = {}

        n_samples, n_features = X.shape

        for feature_index in range(n_features):
            values_of_features = X[:, feature_index]
            thresholds_possibilities = np.unique(values_of_features)

            for threshold in thresholds_possibilities:
                indices_left = values_of_features <= threshold
                indices_right = values_of_features > threshold

                if np.sum(indices_left) == 0 or np.sum(indices_right) == 0:
                    continue

                left_y = y[indices_left]
                right_y = y[indices_right]

                gain_info = self._information_gain(y, left_y, right_y)

                if gain_info > best_gain_info:
                    best_gain_info = gain_info
                    best_split = {
                        'feature_index': feature_index,
                        'threshold': threshold,
                        'left_X': X[indices_left],
                        'left_y': left_y,
                        'right_X': X[indices_right],
                        'right_y': right_y
                    }

        return best_split
    
    def _information_gain(self, y, left_y, right_y):
        parent_entropy = self._entropy(y)
        left_entropy = self._entropy(left_y)
        right_entropy = self._entropy(right_y)

        left_weight = len(left_y) / len(y)
        right_weight = len(right_y) / len(y)
        
        return parent_entropy - (left_weight * left_entropy + right_weight * right_entropy)

    def _entropy(self, y):
        class_counts = {}
        for label in y:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        probabilities = [count / len(y) for count in class_counts.values()]
        return -sum(p * self._log2(p) for p in probabilities)

    def _log2(self, x):
        return 0 if x == 0 else math.log(x, 2)

    def _most_common_class(self, y):
        class_counts = {}
        for label in y:
            class_counts[label] = class_counts.get(label, 0) + 1
        return max(class_counts, key=lambda k: class_counts[k])

    def _predict_one(self, x, tree):
        if 'label' in tree:
            return tree['label']
        feature_value = x[tree['feature_index']]
        if feature_value <= tree['threshold']:
            return self._predict_one(x, tree['left'])
        else:
            return self._predict_one(x, tree['right'])

    def predict(self, X):
        if self.tree is None:
            return np.zeros(X.shape[0])
        return np.array([self._predict_one(x, self.tree) for x in X])


# ============================================================================
# 3. TRAINING PIPELINE
# ============================================================================

def main():
    print("=" * 70)
    print("TRAINING ALL MODELS FOR GUI")
    print("=" * 70)
    
    # Load data
    print("\n[1/5] Loading and preprocessing data...")
    X, y = load_and_preprocess_data()
    
    # Split data
    print("[2/5] Splitting data (80/20 train/test split)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features (the continuous numerical features)
    print("[3/5] Scaling features...")
    SCALING_FEATURES = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi', 'pulse_pressure']
    
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # Fit scaler on RAW training data
    X_train_scaled[SCALING_FEATURES] = scaler.fit_transform(X_train[SCALING_FEATURES])
    X_test_scaled[SCALING_FEATURES] = scaler.transform(X_test[SCALING_FEATURES])
    
    # Save scaler for GUI to use on new raw inputs
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("   Scaler saved to models/scaler.pkl")
    
    # Convert to numpy - data is now properly scaled
    X_train_np = X_train_scaled.values.astype(np.float64)
    X_test_np = X_test_scaled.values.astype(np.float64)
    y_train_np = y_train.values.astype(np.int64)
    y_test_np = y_test.values.astype(np.int64)
    
    # Train models
    print("\n[4/5] Training models...")
    
    # 1. Logistic Regression
    print("\n   Training Logistic Regression...")
    lr_model = LogisticRegressionScratch(lr=0.01, n_iter=2000, l2=0.1, print_every=500)
    lr_model.fit(X_train_np, y_train_np)
    
    with open('models/lr_model.pkl', 'wb') as f:
        pickle.dump(lr_model, f)
    print("   Logistic Regression saved to models/lr_model.pkl")
    
    # Evaluate
    y_pred_lr = lr_model.predict(X_test_np)
    print(f"      Accuracy: {accuracy_score(y_test_np, y_pred_lr):.4f}")
    print(f"      F1-Score: {f1_score(y_test_np, y_pred_lr):.4f}")
    
    # 2. Decision Tree
    print("\n   Training Decision Tree...")
    dt_model = DecisionTree(max_depth=10)
    dt_model.fit(X_train_np, y_train_np)
    
    with open('models/dt_model.pkl', 'wb') as f:
        pickle.dump(dt_model, f)
    print("   Decision Tree saved to models/dt_model.pkl")
    
    # Evaluate
    y_pred_dt = dt_model.predict(X_test_np)
    print(f"      Accuracy: {accuracy_score(y_test_np, y_pred_dt):.4f}")
    print(f"      F1-Score: {f1_score(y_test_np, y_pred_dt):.4f}")
    
    # 3. Random Forest
    print("\n   Training Random Forest...")
    rf_model = CustomRandomForest(n_estimators=100, max_depth=12, random_state=42)
    rf_model.fit(X_train_np, y_train_np)
    
    with open('models/rf_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    print("   Random Forest saved to models/rf_model.pkl")
    
    # Evaluate
    y_pred_rf = rf_model.predict(X_test_np)
    print(f"      Accuracy: {accuracy_score(y_test_np, y_pred_rf):.4f}")
    print(f"      F1-Score: {f1_score(y_test_np, y_pred_rf):.4f}")

    # 4. Support Vector Machine (SVM)
    print("\n   Training SVM (LinearSVC calibrated)...")
    svm_model = CustomSVM(random_state=42, max_iter=2000)
    svm_model.fit(X_train_np, y_train_np)
    
    with open('models/svm_model.pkl', 'wb') as f:
        pickle.dump(svm_model, f)
    print("   SVM saved to models/svm_model.pkl")
    
    # Evaluate
    y_pred_svm = svm_model.predict(X_test_np)
    print(f"      Accuracy: {accuracy_score(y_test_np, y_pred_svm):.4f}")
    print(f"      F1-Score: {f1_score(y_test_np, y_pred_svm):.4f}")
    
    print("\n" + "=" * 70)
    print("[5/5] ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
    print("=" * 70)
    print("\nYou can now run the GUI with:")
    print("  .venv/bin/streamlit run gui.py")
    print("\nThe following files were created in the models/ directory:")
    print("  - scaler.pkl")
    print("  - lr_model.pkl")
    print("  - dt_model.pkl")
    print("  - rf_model.pkl")
    print("  - svm_model.pkl")

if __name__ == "__main__":
    main()
