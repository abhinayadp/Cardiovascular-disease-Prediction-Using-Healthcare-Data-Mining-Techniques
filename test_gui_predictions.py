"""
Automated GUI Prediction Tester
Tests the models with known cases to verify predictions are correct
"""
import pickle
import numpy as np
import pandas as pd
import math
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field

# Model class definitions (needed for unpickling)
def _sigmoid(z: np.ndarray) -> np.ndarray:
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
    
    def fit(self, X, y):
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
    
    def fit(self, X, y):
        return self
    
    def _euclidean_distance(self, x1, x2):
        return float(np.linalg.norm(x1 - x2))
    
    def _predict(self, x_new):
        distances = [self._euclidean_distance(x_new, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return int(most_common[0][0])
    
    def predict(self, X):
        return np.array([self._predict(x) for x in X])

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None
    
    def fit(self, X, y):
        return self
    
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

# Load models and scaler
print("Loading models and scaler...")
with open('models/lr_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)
with open('models/dt_model.pkl', 'rb') as f:
    dt_model = pickle.load(f)
with open('models/knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

models = {
    'Logistic Regression': lr_model,
    'Decision Tree': dt_model,
    'K-NN': knn_model
}

# Define test cases
test_cases = [
    {
        'name': 'HIGH RISK - Elderly with Multiple Risks',
        'expected': 'HIGH RISK',
        'input': {
            'age': 65, 'height': 165, 'weight': 95,
            'ap_hi': 180, 'ap_lo': 110,
            'gender': 2, 'cholesterol': 3, 'gluc': 3,
            'smoke': 1, 'alco': 1, 'active': 0
        }
    },
    {
        'name': 'LOW RISK - Healthy Young Adult',
        'expected': 'LOW RISK',
        'input': {
            'age': 35, 'height': 175, 'weight': 70,
            'ap_hi': 115, 'ap_lo': 75,
            'gender': 1, 'cholesterol': 1, 'gluc': 1,
            'smoke': 0, 'alco': 0, 'active': 1
        }
    },
    {
        'name': 'MODERATE-HIGH RISK - Obese with Hypertension',
        'expected': 'HIGH RISK',
        'input': {
            'age': 58, 'height': 170, 'weight': 88,
            'ap_hi': 145, 'ap_lo': 95,
            'gender': 1, 'cholesterol': 2, 'gluc': 2,
            'smoke': 0, 'alco': 1, 'active': 1
        }
    },
    {
        'name': 'LOW RISK - Young Female Athlete',
        'expected': 'LOW RISK',
        'input': {
            'age': 28, 'height': 168, 'weight': 62,
            'ap_hi': 110, 'ap_lo': 70,
            'gender': 2, 'cholesterol': 1, 'gluc': 1,
            'smoke': 0, 'alco': 0, 'active': 1
        }
    },
    {
        'name': 'BORDERLINE - Middle Age Overweight',
        'expected': 'BORDERLINE',
        'input': {
            'age': 50, 'height': 172, 'weight': 78,
            'ap_hi': 135, 'ap_lo': 85,
            'gender': 1, 'cholesterol': 2, 'gluc': 1,
            'smoke': 0, 'alco': 0, 'active': 1
        }
    }
]

def preprocess_input(input_data, scaler):
    """Preprocess input exactly like the GUI does."""
    # Calculate BMI and pulse pressure
    bmi = input_data['weight'] / (input_data['height'] / 100) ** 2
    pulse_pressure = input_data['ap_hi'] - input_data['ap_lo']
    
    # Features for scaling
    scaling_features = np.array([[
        input_data['age'],
        input_data['height'],
        input_data['weight'],
        input_data['ap_hi'],
        input_data['ap_lo'],
        bmi,
        pulse_pressure
    ]])
    
    # Scale
    scaled = scaler.transform(scaling_features)
    
    # Add categorical features
    categorical = np.array([
        input_data['gender'],
        input_data['cholesterol'],
        input_data['gluc'],
        input_data['smoke'],
        input_data['alco'],
        input_data['active']
    ])
    
    # Combine
    final_input = np.concatenate([scaled.flatten(), categorical]).reshape(1, -1)
    
    return final_input, bmi, pulse_pressure

def test_predictions():
    """Test all models with all test cases."""
    print("=" * 80)
    print("AUTOMATED GUI PREDICTION TESTING")
    print("=" * 80)
    
    total_tests = len(test_cases) * len(models)
    passed_tests = 0
    
    for test_idx, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"TEST CASE {test_idx}: {test_case['name']}")
        print(f"Expected Outcome: {test_case['expected']}")
        print(f"{'='*80}")
        
        # Preprocess input
        input_data = test_case['input']
        scaled_input, bmi, pulse_pressure = preprocess_input(input_data, scaler)
        
        print(f"\nInput Summary:")
        print(f"  Age: {input_data['age']} years")
        print(f"  BMI: {bmi:.1f} (calculated)")
        print(f"  Blood Pressure: {input_data['ap_hi']}/{input_data['ap_lo']} mmHg")
        print(f"  Pulse Pressure: {pulse_pressure} mmHg")
        print(f"  Cholesterol: {input_data['cholesterol']}, Glucose: {input_data['gluc']}")
        
        print(f"\nModel Predictions:")
        print(f"  {'Model':<20} {'Prediction':<15} {'Score/Prob':<15} {'Match?':<10}")
        print(f"  {'-'*60}")
        
        for model_name, model in models.items():
            # Make prediction
            prediction = model.predict(scaled_input)[0]
            
            # Get probability/score if available
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(scaled_input)[0, 1]
                score_str = f"{prob:.3f}"
            else:
                score_str = f"{prediction}"
            
            # Determine risk level
            risk_status = "HIGH RISK" if prediction == 1 else "LOW RISK"
            
            # Check if matches expected (allowing borderline cases)
            if test_case['expected'] == 'BORDERLINE':
                match = "✓ OK"  # Any prediction is acceptable for borderline
                passed_tests += 1
            elif risk_status == test_case['expected']:
                match = "✓ PASS"
                passed_tests += 1
            else:
                match = "✗ FAIL"
            
            print(f"  {model_name:<20} {risk_status:<15} {score_str:<15} {match:<10}")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Tests Passed: {passed_tests}/{total_tests}")
    print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n✓ ALL TESTS PASSED! Models are working correctly.")
    elif passed_tests >= total_tests * 0.8:
        print("\n⚠ MOST TESTS PASSED. Some borderline cases may vary.")
    else:
        print("\n✗ MANY TESTS FAILED. Please review model training.")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR GUI TESTING:")
    print("="*80)
    print("1. Open GUI: .venv/bin/streamlit run gui.py")
    print("2. Test the same cases manually in the GUI")
    print("3. Verify BMI and Pulse Pressure calculations match")
    print("4. Compare predictions with results above")
    print("5. Check all three models (LR, DT, K-NN) give similar results")
    print("\nSee test_cases_for_gui.md for detailed test instructions!")

if __name__ == "__main__":
    test_predictions()
