<div align="center">
  <h1>🩺 Cardiovascular Disease Prediction System</h1>
  <p><strong>An End-to-End Machine Learning Application for Healthcare Data Mining</strong></p>
</div>

---

## Overview

This project is a comprehensive Machine Learning system designed to predict the risk of Cardiovascular Disease (CVD) based on patient health metrics and lifestyle choices. 

This project demonstrates a complete end-to-end data science lifecycle, from robust data preprocessing and feature engineering, to training both library-based and **custom from-scratch algorithms**, culminating in an interactive Streamlit Web GUI for clinical predictions.

## Key Features

*   **Robust Data Preprocessing:** Cleans noisy healthcare data, handles implausible outliers (e.g., negative blood pressure), and prepares reliable datasets.
*   **Intelligent Feature Engineering:** Derives critical clinical features such as **BMI** (from height and weight) and **Pulse Pressure** (Systolic - Diastolic BP) to improve model predictive power.
*   **"From Scratch" Algorithm Implementations:** To demonstrate a deep mathematical understanding of ML, several algorithms (like Logistic Regression, K-Nearest Neighbors, and Decision Trees) were implemented manually using `numpy` alongside standard `scikit-learn` models.
*   **Interactive Clinical GUI:** A sleek, user-friendly web interface built with **Streamlit** that allows a user (or doctor) to input patient metrics and receive instant risk assessments and interpretations.
*   **Model Comparison:** Evaluates and compares multiple algorithms including Logistic Regression, Decision Trees, Random Forests, and Support Vector Machines (SVM).

---

##  Project Structure

```text
 Cardiovascular-Disease-Prediction
 ┣ 📂 Dataset/               # Contains raw and cleaned data (cardio_train.csv)
 ┣ 📂 Models/                # Custom from-scratch algorithm implementations
 ┣ 📂 Preprocessing/         # Data cleaning and feature engineering pipelines
 ┣ 📂 Outputs/               # Generated graphs and exploration visualizations
 ┣ 📜 gui.py                 # The main Streamlit Web Application
 ┣ 📜 train_and_save_models.py # Script to train and persist models (.pkl)
 ┣ 📜 evaluate_models.py     # Script to test and evaluate model accuracy
 ┣ 📜 main.py                # Entry point for the raw preprocessing flow
 ┗ 📜 requirements.txt       # Project dependencies
```

---

## Installation & Setup

It is highly recommended to use a virtual environment to run this project. 

**1. Clone the repository**
```bash
git clone https://github.com/abhinayadp/Cardiovascular-disease-Prediction-Using-Healthcare-Data-Mining-Techniques.git
cd Cardiovascular-disease-Prediction-Using-Healthcare-Data-Mining-Techniques
```

**2. Create and activate a Virtual Environment**
```bash
# On Windows
python -m venv .venv
.\.venv\Scripts\activate

# On Mac/Linux
python -m venv .venv
source .venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## Usage Instructions

### 1. Launch the Web Application (GUI)
The easiest way to interact with the models is through the Streamlit interface. Make sure your virtual environment is active, then run:

```bash
streamlit run gui.py
```
This will open a web browser at `http://localhost:8501` where you can input patient data and generate predictions.

### 2. Retrain the Models
If you wish to run the preprocessing pipeline and retrain the models from scratch (this will output new `.pkl` files to the `models/` directory):

```bash
python main.py
python train_and_save_models.py
```

---

## Under the Hood: The Models

This project implements several approaches to classify patient risk:

1.  **Logistic Regression:** Excellent for clinical interpretability. It evaluates the linear relationship between our features (like Age and Pulse Pressure) and the log-odds of having CVD.
2.  **Decision Tree:** Provides a clear "clinical pathway" or ruleset (e.g., *If Age > 55 and BMI > 30, then High Risk*).
3.  **Random Forest:** An ensemble method that averages the decisions of hundreds of trees, protecting against overfitting and improving general accuracy.
4.  **Support Vector Machine (SVM):** Finds the optimal hyperplane in our 13-dimensional feature space to distinctly separate high-risk and low-risk patients.
5.  **K-Nearest Neighbors (KNN):** A custom from-scratch implementation that classifies patients based on their similarity to historical cases. *(Note: While successfully implemented, this model is excluded from the active Streamlit GUI as it proved less reliable/stable for live clinical predictions compared to the others).*

*Note: All numerical features are scaled using `StandardScaler` to ensure algorithms relying on distance metrics (like SVM and K-NN) or gradients (like Logistic Regression) perform optimally.*

---

## Contributing
Feel free to fork this repository, submit pull requests, or open an issue if you have suggestions for new features or improvements.
