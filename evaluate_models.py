"""
Comprehensive Model Evaluation Script
Evaluates all trained models and provides detailed performance metrics
"""
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import math
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# MODEL CLASS DEFINITIONS (needed for unpickling)


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
        return {}

    def _best_split(self, X, y):
        return {}
    
    def _information_gain(self, y, left_y, right_y):
        return 0

    def _entropy(self, y):
        return 0

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

def load_data():
    """Load and prepare the dataset."""
    print("Loading dataset...")
    # Load RAW data (same as training)
    df = pd.read_csv('Dataset/cardio_train.csv', sep=';')
    
    # Convert age from days to years
    df['age'] = (df['age'] / 365.25).astype(int)
    
    # Basic data cleaning
    swapped = df['ap_hi'] < df['ap_lo']
    df.loc[swapped, ['ap_hi', 'ap_lo']] = df.loc[swapped, ['ap_lo', 'ap_hi']].values
    
    # Remove outliers
    df = df[(df['age'] >= 30) & (df['age'] <= 80)]
    df = df[(df['height'] >= 130) & (df['height'] <= 210)]
    df = df[(df['weight'] >= 40) & (df['weight'] <= 200)]
    df = df[(df['ap_hi'] >= 80) & (df['ap_hi'] <= 240)]
    df = df[(df['ap_lo'] >= 40) & (df['ap_lo'] <= 150)]
    
    df = df.drop_duplicates(ignore_index=True)
    
    # Feature Engineering
    df['bmi'] = df['weight'] / (df['height'] / 100) ** 2
    df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
    
    # Drop unnecessary columns
    cols_to_drop = ['cardio', 'id']
    X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    y = df['cardio']
    
    return X, y

def load_models():
    """Load all trained models."""
    print("Loading models...")
    models = {}
    model_files = {
        'Logistic Regression': 'models/lr_model.pkl',
        'Decision Tree': 'models/dt_model.pkl',
    }
    
    for name, filepath in model_files.items():
        try:
            with open(filepath, 'rb') as f:
                models[name] = pickle.load(f)
            print(f"  {name} loaded")
        except FileNotFoundError:
            print(f"  {name} not found at {filepath}")
    
    return models

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a single model and return metrics."""
    print(f"\nEvaluating {model_name}...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0),
        'Specificity': None,  # Will calculate from confusion matrix
    }
    
    # Calculate ROC-AUC if model has predict_proba
    try:
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
            metrics['ROC-AUC'] = roc_auc_score(y_test, y_proba)
        else:
            metrics['ROC-AUC'] = roc_auc_score(y_test, y_pred)
    except:
        metrics['ROC-AUC'] = None
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate Specificity (True Negative Rate)
    metrics['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Additional useful metrics
    metrics['True Positives'] = tp
    metrics['True Negatives'] = tn
    metrics['False Positives'] = fp
    metrics['False Negatives'] = fn
    
    return metrics, cm, y_pred

def plot_confusion_matrices(cms, model_names):
    """Plot confusion matrices for all models."""
    fig, axes = plt.subplots(1, len(cms), figsize=(15, 4))
    
    if len(cms) == 1:
        axes = [axes]
    
    for idx, (cm, name) in enumerate(zip(cms, model_names)):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['No CVD', 'CVD'], yticklabels=['No CVD', 'CVD'])
        axes[idx].set_title(f'{name}\nConfusion Matrix')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('Dataset/figures/confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("  Confusion matrices saved to Dataset/figures/confusion_matrices.png")
    plt.close()

def plot_roc_curves(models, X_test, y_test):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_proba = model.predict(X_test)
            
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', linewidth=2)
        except Exception as e:
            print(f"  ! Could not plot ROC for {name}: {e}")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    
    plt.savefig('Dataset/figures/roc_curves.png', dpi=300, bbox_inches='tight')
    print("  ROC curves saved to Dataset/figures/roc_curves.png")
    plt.close()

def plot_metrics_comparison(metrics_df):
    """Plot comparison of key metrics across models."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for idx, metric in enumerate(metrics_to_plot):
        row = idx // 2
        col = idx % 2
        
        ax = axes[row, col]
        values = metrics_df[metric].values
        bars = ax.bar(metrics_df['Model'], values, color=colors[:len(metrics_df)])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel(metric, fontsize=11, fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=15)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Dataset/figures/metrics_comparison.png', dpi=300, bbox_inches='tight')
    print("  Metrics comparison saved to Dataset/figures/metrics_comparison.png")
    plt.close()

def interpret_results(metrics_df):
    """Provide interpretation of model performance."""
    print("\n" + "="*80)
    print("MODEL PERFORMANCE INTERPRETATION")
    print("="*80)
    
    # Find best model for each metric
    best_accuracy = metrics_df.loc[metrics_df['Accuracy'].idxmax()]
    best_f1 = metrics_df.loc[metrics_df['F1-Score'].idxmax()]
    best_recall = metrics_df.loc[metrics_df['Recall'].idxmax()]
    best_precision = metrics_df.loc[metrics_df['Precision'].idxmax()]
    
    print(f"\nBEST PERFORMERS:")
    print(f"  Highest Accuracy:  {best_accuracy['Model']} ({best_accuracy['Accuracy']:.3f})")
    print(f"  Highest F1-Score:  {best_f1['Model']} ({best_f1['F1-Score']:.3f})")
    print(f"  Highest Recall:    {best_recall['Model']} ({best_recall['Recall']:.3f})")
    print(f"  Highest Precision: {best_precision['Model']} ({best_precision['Precision']:.3f})")
    
    print(f"\nPERFORMANCE ASSESSMENT:")
    
    for _, row in metrics_df.iterrows():
        model = row['Model']
        acc = row['Accuracy']
        f1 = row['F1-Score']
        
        print(f"\n{model}:")
        
        # Accuracy assessment
        if acc >= 0.80:
            print(f"  EXCELLENT accuracy ({acc:.1%}) - Model performs very well")
        elif acc >= 0.70:
            print(f"  GOOD accuracy ({acc:.1%}) - Model performs well")
        elif acc >= 0.60:
            print(f"  FAIR accuracy ({acc:.1%}) - Room for improvement")
        else:
            print(f"  POOR accuracy ({acc:.1%}) - Needs significant improvement")
        
        # Precision vs Recall balance
        prec = row['Precision']
        rec = row['Recall']
        
        if abs(prec - rec) < 0.05:
            print(f"  Well-balanced precision ({prec:.3f}) and recall ({rec:.3f})")
        elif prec > rec:
            print(f"  Higher precision ({prec:.3f}) than recall ({rec:.3f})")
            print(f"     Model is conservative (fewer false positives, more false negatives)")
        else:
            print(f"  Higher recall ({rec:.3f}) than precision ({prec:.3f})")
            print(f"     Model is aggressive (more false positives, fewer false negatives)")
        
        # Clinical context
        print(f"  For CVD screening: ", end="")
        if rec >= 0.75:
            print(f"Good recall ({rec:.3f}) - catches most disease cases")
        else:
            print(f"Low recall ({rec:.3f}) - may miss disease cases")

def save_detailed_report(metrics_df, output_path='Dataset/figures/model_evaluation_report.txt'):
    """Save detailed evaluation report to file."""
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CARDIOVASCULAR DISEASE PREDICTION - MODEL EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("SUMMARY METRICS\n")
        f.write("-"*80 + "\n")
        f.write(metrics_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']].to_string(index=False))
        f.write("\n\n")
        
        f.write("DETAILED CLASSIFICATION METRICS\n")
        f.write("-"*80 + "\n")
        for _, row in metrics_df.iterrows():
            f.write(f"\n{row['Model']}:\n")
            f.write(f"  True Positives:  {row['True Positives']}\n")
            f.write(f"  True Negatives:  {row['True Negatives']}\n")
            f.write(f"  False Positives: {row['False Positives']}\n")
            f.write(f"  False Negatives: {row['False Negatives']}\n")
            f.write(f"  Specificity:     {row['Specificity']:.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("INTERPRETATION GUIDE\n")
        f.write("="*80 + "\n")
        f.write("""
Accuracy:  Overall correctness (TP+TN)/(TP+TN+FP+FN)
Precision: Of predicted positives, how many are correct? TP/(TP+FP)
Recall:    Of actual positives, how many did we catch? TP/(TP+FN)
F1-Score:  Harmonic mean of precision and recall
ROC-AUC:   Area under ROC curve (0.5=random, 1.0=perfect)
Specificity: Of actual negatives, how many did we correctly identify? TN/(TN+FP)

For Medical Diagnosis:
- High Recall is crucial (don't miss disease cases)
- Balance with Precision to avoid too many false alarms
- F1-Score gives overall balance
""")
    
    print(f"  Detailed report saved to {output_path}")

def main():
    print("="*80)
    print("CARDIOVASCULAR DISEASE PREDICTION - MODEL EVALUATION")
    print("="*80)
    
    # Create output directory
    Path("Dataset/figures").mkdir(parents=True, exist_ok=True)
    
    # Load data and models
    X, y = load_data()
    models = load_models()
    
    if not models:
        print("\n No models found! Please train models first using train_and_save_models.py")
        return
    
    # Split data (same as training)
    print("\nSplitting data for evaluation...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the data like in training
    from sklearn.preprocessing import StandardScaler
    SCALING_FEATURES = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi', 'pulse_pressure']
    
    scaler = StandardScaler()
    X_test_scaled = X_test.copy()
    X_test_scaled[SCALING_FEATURES] = scaler.fit_transform(X_train[SCALING_FEATURES]).mean(axis=0)  # dummy
    
    # Actually use the saved scaler
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    X_test_scaled[SCALING_FEATURES] = scaler.transform(X_test[SCALING_FEATURES])
    
    X_test_np = X_test_scaled.values.astype(np.float64)
    y_test_np = y_test.values.astype(np.int64)
    
    print(f"Test set size: {len(X_test)} samples")
    
    # Evaluate each model
    all_metrics = []
    all_cms = []
    model_names = []
    
    for name, model in models.items():
        metrics, cm, y_pred = evaluate_model(model, X_test_np, y_test_np, name)
        all_metrics.append(metrics)
        all_cms.append(cm)
        model_names.append(name)
    
    # Create DataFrame
    metrics_df = pd.DataFrame(all_metrics)
    
    # Display results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print("\n" + metrics_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']].to_string(index=False))
    
    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    plot_confusion_matrices(all_cms, model_names)
    plot_roc_curves(models, X_test_np, y_test_np)
    plot_metrics_comparison(metrics_df)
    
    # Save detailed report
    save_detailed_report(metrics_df)
    
    # Interpret results
    interpret_results(metrics_df)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  Dataset/figures/confusion_matrices.png")
    print("  Dataset/figures/roc_curves.png")
    print("  Dataset/figures/metrics_comparison.png")
    print("  Dataset/figures/model_evaluation_report.txt")

if __name__ == "__main__":
    main()
