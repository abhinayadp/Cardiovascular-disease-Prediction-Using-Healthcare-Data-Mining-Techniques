import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import time

print("Loading data...")
df = pd.read_csv('Dataset/cardio_train.csv', sep=';')
df['age'] = (df['age'] / 365.25).astype(int)
swapped = df['ap_hi'] < df['ap_lo']
df.loc[swapped, ['ap_hi', 'ap_lo']] = df.loc[swapped, ['ap_lo', 'ap_hi']].values
df = df[(df['age'] >= 30) & (df['age'] <= 80)]
df = df[(df['height'] >= 130) & (df['height'] <= 210)]
df = df[(df['weight'] >= 40) & (df['weight'] <= 200)]
df = df[(df['ap_hi'] >= 80) & (df['ap_hi'] <= 240)]
df = df[(df['ap_lo'] >= 40) & (df['ap_lo'] <= 150)]
df = df.drop_duplicates(ignore_index=True)
df['bmi'] = df['weight'] / (df['height'] / 100) ** 2
df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
X = df.drop(columns=['cardio', 'id'])
y = df['cardio']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
SCALING_FEATURES = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi', 'pulse_pressure']
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[SCALING_FEATURES] = scaler.fit_transform(X_train[SCALING_FEATURES])
X_test_scaled[SCALING_FEATURES] = scaler.transform(X_test[SCALING_FEATURES])

print("Training Random Forest...")
start = time.time()
rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
print(f"RF Time: {time.time()-start:.2f}s, Acc: {accuracy_score(y_test, y_pred_rf):.4f}, F1: {f1_score(y_test, y_pred_rf):.4f}")

print("Training SVM (subset 10k for speed)...")
start = time.time()
svm = SVC(probability=True, random_state=42, max_iter=2000)
svm.fit(X_train_scaled[:10000], y_train[:10000])
y_pred_svm = svm.predict(X_test_scaled)
print(f"SVM Time: {time.time()-start:.2f}s, Acc: {accuracy_score(y_test, y_pred_svm):.4f}, F1: {f1_score(y_test, y_pred_svm):.4f}")
