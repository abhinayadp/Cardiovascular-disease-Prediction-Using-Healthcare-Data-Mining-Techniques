# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
import pickle
import math
from pathlib import Path
from collections import Counter
from sklearn.metrics import roc_auc_score
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
import warnings

# CUSTOM MODEL DEFINITIONS

# A. Logistic Regression (lr_model.py)
def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))

@dataclass
class LogisticRegressionScratch:
    # Hyperparameters
    lr: float = 0.001          # learning rate
    n_iter: int = 8000         # max gradient steps
    l2: float = 0.0            # L2 regularization strength
    random_state: int | None = 42
    print_every: int = 400     # how often to print loss

    # Learned params
    w: np.ndarray = field(init=False)
    b: float = field(init=False, default=0.0)

    def fit(self, X: np.ndarray, y: np.ndarray):
        n, d = X.shape
        self.w = np.zeros(d, dtype=float)
        self.b = 0.0
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(y=1|x) for each row in X."""
        if not hasattr(self, 'w'):
            raise Exception("Model not fitted/loaded correctly.")
            
        # Returns a 2D array [P(y=0), P(y=1)]
        proba_1 = _sigmoid(X @ self.w + self.b)
        proba_0 = 1.0 - proba_1
        return np.column_stack((proba_0, proba_1))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return hard labels using the given decision threshold."""
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

# B. Custom K-NN Classifier (knn_model.py)
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

# C. Decision Tree Classifier (dt_model.py)
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)
        return self

    def _log2(self, x): return 0 if x == 0 else math.log(x, 2)
    def _entropy(self, y): 
        if len(y) == 0: return 0
        probs = np.bincount(y.astype(int)) / len(y)
        return -sum(p * self._log2(p) for p in probs if p > 0)
    def _most_common_class(self, y): 
        return Counter(y).most_common(1)[0][0]
    def _information_gain(self, y, left_y, right_y): return 0 
    def _best_split(self, X, y): return {} 
    def _build_tree(self, X, y, depth=0): return {} 
    
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
            return np.zeros(X.shape[0]) # Return 0s if not loaded
        return np.array([self._predict_one(x, self.tree) for x in X])

# CORE APPLICATION LOGIc

MODEL_OPTIONS = {
    "Logistic Regression (LR)": "lr_model.pkl",
    "Decision Tree (DT)": "dt_model.pkl",
    "Random Forest (RF)": "rf_model.pkl",
    "Support Vector Machine (SVM)": "svm_model.pkl",
}
SCALER_PATH = Path("models/scaler.pkl")


@st.cache_resource
def load_models_and_scaler() -> Tuple[Optional[Dict], Optional[Any]]:
    """Loads all saved model files (.pkl) and the scaler from the 'models/' folder."""
    models = {}
    scaler = None
    
    try:
        # load the critical StandardScaler object first
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
            
        # load the three trained models
        for name, filename in MODEL_OPTIONS.items():
            with open(Path("models") / filename, 'rb') as f:
                models[name] = pickle.load(f)
                
        return models, scaler
        
    except FileNotFoundError:
        st.error("FATAL ERROR: Model files not found. Ensure models/ directory exists and contains all required files.")
        return None, None
    except Exception as e:
        st.error(f"FATAL ERROR during model loading: {e}")
        return None, None

def preprocess_input(input_data: Dict[str, Any], scaler: Any) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """Transforms user input into the scaled feature vector, adjusted for the 7-feature scaler."""
    try:
        # Feature Engineering (BMI and Pulse Pressure)
        df_input = pd.DataFrame([input_data])
        df_input['bmi'] = df_input['weight'] / (df_input['height'] / 100)**2
        df_input['pulse_pressure'] = df_input['ap_hi'] - df_input['ap_lo']
        
        # FINAL FEATURE LIST FOR SCALING
        SCALING_ONLY_FEATURES = [
            'age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi', 'pulse_pressure'
        ]
        
        # Scale the continuous features using the fitted scaler
        df_input[SCALING_ONLY_FEATURES] = scaler.transform(df_input[SCALING_ONLY_FEATURES])

        # Arrange columns in the exact order the model expects (same as training data)
        EXPECTED_COLUMNS = [
            'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 
            'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi', 'pulse_pressure'
        ]
        
        final_13_feature_vector = df_input[EXPECTED_COLUMNS].values

        
        return final_13_feature_vector, df_input['bmi'].iloc[0] 

    except Exception as e:
        st.error(f"Error during input preprocessing. Details: {e}")
        return None, None


# STREAMLIT GUI LAYOUT

def main():
    st.set_page_config(layout="wide")
    st.title("Cardiovascular Disease Risk Prediction 🩺")
    # Load models and scaler 
    models, scaler = load_models_and_scaler()

    if models is None or scaler is None:
        st.stop()
    
    # Type assertion for type checker
    assert models is not None
    assert scaler is not None

    # GUI Layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Model Selection & Input")
        
        # Model Selection 
        selected_model_name = st.radio(
            "Choose a Classifier:",
            list(MODEL_OPTIONS.keys()),
            key="model_select",
            index=1,  # Default to Decision Tree 
        )
        prediction_model = models.get(selected_model_name)
        
        if prediction_model is None:
            st.error("Model not found")
            st.stop()
        
        st.markdown("---")
        st.subheader("Patient Input")
        
        # Input Features
        input_data = {}
        input_col_1, input_col_2 = st.columns(2)

        # Numerical and Continuous Inputs
        with input_col_1:
            input_data['age'] = st.number_input("Age (years)", min_value=18, max_value=100, value=55)
            input_data['height'] = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
            input_data['weight'] = st.number_input("Weight (kg)", min_value=30, max_value=300, value=75)
            input_data['ap_hi'] = st.number_input("Systolic BP (ap_hi)", min_value=80, max_value=240, value=120)
            input_data['ap_lo'] = st.number_input("Diastolic BP (ap_lo)", min_value=40, max_value=150, value=80)
        
        # Categorical and Binary Inputs
        with input_col_2:
            input_data['gender'] = st.selectbox("Gender (1=M, 2=F)", options=[1, 2])
            input_data['cholesterol'] = st.selectbox("Cholesterol Level (1-3)", options=[1, 2, 3])
            input_data['gluc'] = st.selectbox("Glucose Level (1-3)", options=[1, 2, 3])
            input_data['smoke'] = 1 if st.checkbox("Smokes", value=False) else 0
            input_data['alco'] = 1 if st.checkbox("Drinks Alcohol", value=False) else 0
            input_data['active'] = 1 if st.checkbox("Physically Active", value=True) else 0

        st.markdown("---")
        
        # Prediction Button
        if st.button("Run Prediction", type="primary"):
            st.session_state['run_prediction'] = True
        
        if st.button("Clear Results"):
            st.session_state['run_prediction'] = False
            st.rerun()
    
    with col2:
        st.header("Prediction & Results")
        
        # Initial check to run prediction
        if st.session_state.get('run_prediction', False):
            # Get the selected model again within this scope
            selected_model = models.get(selected_model_name)
            if selected_model is None:
                st.error("Selected model could not be loaded")
                st.stop()
                
            scaled_input, bmi = preprocess_input(input_data, scaler)
            
            if scaled_input is not None:
                # Run the prediction
                prediction = selected_model.predict(scaled_input)[0]  # type: ignore
                
                # Try to get probability
                score = 0.5 
                try:
                    if selected_model_name == "Logistic Regression (LR)":
                        score = selected_model.predict_proba(scaled_input)[0, 1]  # type: ignore
                    else:
                        # For Decision Tree, use prediction as score
                        score = prediction  # type: ignore
                except:
                    score = prediction 
                
                # Display Prediction
                risk_status = "HIGH RISK (CVD Detected)" if prediction == 1 else "LOW RISK (No CVD)"
                color = "red" if prediction == 1 else "green"
                
                st.markdown(f"## Predicted Outcome: <span style='color:{color};'>{risk_status}</span>", unsafe_allow_html=True)
                st.markdown(f"**Score / Likelihood of CVD (P=1):** **{score:.2f}**")
                
                st.markdown("---")
                
                
                # Detailed Interpretation 
                if selected_model_name == "Logistic Regression (LR)":
                    st.success("This model is highly valuable for interpretation.")
                    if score > 0.5:
                        st.write("The **high positive coefficients** (from training) for **Age, Systolic BP, and BMI** drove the prediction toward high risk.")
                    else:
                        st.write("The model found that key risk factors were within normal ranges or balanced by lifestyle (active/non-smoking).")

                elif selected_model_name == "Decision Tree (DT)":
                    st.success("The DT provides a clear clinical pathway.")
                    st.write("The prediction followed a specific **branch of clinical rules** (e.g., *IF Age > 55 AND BMI > 30*).")
                    st.write("This makes the result directly defensible using simple thresholds.")

                elif selected_model_name == "Random Forest (RF)":
                    st.success("RF provides highly reliable consensus predictions.")
                    st.write("This prediction is the **majority vote of hundreds of distinct decision trees**.")
                    st.write("By averaging many trees, it minimizes the errors and biases of any single pathway.")

                elif selected_model_name == "Support Vector Machine (SVM)":
                    st.success("SVM separates high and low risk patients mathematically.")
                    st.write("The model looked at how your metrics fall relative to the **maximum-margin boundary** in 13-dimensional space.")
                    st.write("Scores closer to 1.0 indicate you are deeply within the high-risk region.")

                st.markdown("---")
                
                st.subheader("Calculated Patient Metrics:")
                st.markdown(f"**Calculated BMI:** **{bmi:.2f}**")
                st.markdown(f"**Pulse Pressure:** {input_data['ap_hi'] - input_data['ap_lo']} mmHg")


if __name__ == '__main__':
    main()