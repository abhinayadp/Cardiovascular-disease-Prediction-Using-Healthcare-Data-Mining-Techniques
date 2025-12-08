# Test Cases for GUI Validation

Use these test cases in your GUI to verify predictions are working correctly.

## Test Case 1: HIGH RISK Patient ⚠️
**Expected Prediction: HIGH RISK (CVD Detected)**

### Input Parameters:
- **Age**: 65 years
- **Height**: 165 cm
- **Weight**: 95 kg (→ BMI ≈ 34.9)
- **Systolic BP (ap_hi)**: 180 mmHg
- **Diastolic BP (ap_lo)**: 110 mmHg
- **Gender**: 2 (Female)
- **Cholesterol**: 3 (High)
- **Glucose**: 3 (High)
- **Smokes**: ✓ Yes
- **Drinks Alcohol**: ✓ Yes
- **Physically Active**: ✗ No

**Why HIGH RISK?**
- Obesity (BMI 34.9)
- High blood pressure (Stage 2 Hypertension)
- High cholesterol and glucose
- Smoking and alcohol use
- Sedentary lifestyle
- Older age

---

## Test Case 2: LOW RISK Patient ✓
**Expected Prediction: LOW RISK (No CVD)**

### Input Parameters:
- **Age**: 35 years
- **Height**: 175 cm
- **Weight**: 70 kg (→ BMI ≈ 22.9)
- **Systolic BP (ap_hi)**: 115 mmHg
- **Diastolic BP (ap_lo)**: 75 mmHg
- **Gender**: 1 (Male)
- **Cholesterol**: 1 (Normal)
- **Glucose**: 1 (Normal)
- **Smokes**: ✗ No
- **Drinks Alcohol**: ✗ No
- **Physically Active**: ✓ Yes

**Why LOW RISK?**
- Young age
- Healthy BMI (22.9)
- Normal blood pressure
- Normal cholesterol and glucose
- Non-smoker, no alcohol
- Physically active

---

## Test Case 3: MODERATE-HIGH RISK Patient ⚠️
**Expected Prediction: HIGH RISK or borderline**

### Input Parameters:
- **Age**: 58 years
- **Height**: 170 cm
- **Weight**: 88 kg (→ BMI ≈ 30.4)
- **Systolic BP (ap_hi)**: 145 mmHg
- **Diastolic BP (ap_lo)**: 95 mmHg
- **Gender**: 1 (Male)
- **Cholesterol**: 2 (Above Normal)
- **Glucose**: 2 (Above Normal)
- **Smokes**: ✗ No
- **Drinks Alcohol**: ✓ Yes
- **Physically Active**: ✓ Yes

**Why MODERATE-HIGH RISK?**
- Older age (58)
- Obesity (BMI 30.4)
- Elevated blood pressure (Stage 1 Hypertension)
- Elevated cholesterol and glucose
- Mixed lifestyle (active but drinks)

---

## Test Case 4: LOW RISK Young Adult ✓
**Expected Prediction: LOW RISK**

### Input Parameters:
- **Age**: 28 years
- **Height**: 168 cm
- **Weight**: 62 kg (→ BMI ≈ 22.0)
- **Systolic BP (ap_hi)**: 110 mmHg
- **Diastolic BP (ap_lo)**: 70 mmHg
- **Gender**: 2 (Female)
- **Cholesterol**: 1 (Normal)
- **Glucose**: 1 (Normal)
- **Smokes**: ✗ No
- **Drinks Alcohol**: ✗ No
- **Physically Active**: ✓ Yes

**Why LOW RISK?**
- Very young
- Excellent BMI
- Excellent blood pressure
- All health markers normal
- Healthy lifestyle

---

## Test Case 5: MODERATE RISK Patient (Borderline) 
**Expected Prediction: Could be either (test model sensitivity)**

### Input Parameters:
- **Age**: 50 years
- **Height**: 172 cm
- **Weight**: 78 kg (→ BMI ≈ 26.4)
- **Systolic BP (ap_hi)**: 135 mmHg
- **Diastolic BP (ap_lo)**: 85 mmHg
- **Gender**: 1 (Male)
- **Cholesterol**: 2 (Above Normal)
- **Glucose**: 1 (Normal)
- **Smokes**: ✗ No
- **Drinks Alcohol**: ✗ No
- **Physically Active**: ✓ Yes

**Why BORDERLINE?**
- Middle age (50)
- Slightly overweight (BMI 26.4)
- Prehypertension (135/85)
- Slightly elevated cholesterol
- Good lifestyle habits

---

## How to Test:

1. **Start the GUI**: `.venv/bin/streamlit run gui.py`
2. **Test each model** (LR, DT, K-NN) with the same inputs
3. **Compare predictions** across models
4. **Check that**:
   - Test Cases 1 & 3 → HIGH RISK
   - Test Cases 2 & 4 → LOW RISK
   - Test Case 5 → May vary (this tests decision boundary)

## What to Look For:

✅ **Good Signs:**
- All 3 models agree on clear cases (1, 2, 4)
- BMI and Pulse Pressure calculated correctly
- Scores/probabilities make sense (higher for HIGH RISK)
- Decision Tree shows ~0.7-0.8 probability for high-risk cases

⚠️ **Warning Signs:**
- Models completely disagree on clear cases
- All predictions are the same regardless of input
- Calculated BMI is wrong
- Scores don't match predictions (e.g., 0.9 probability but says LOW RISK)

## Expected Calculated Values:

| Test Case | BMI | Pulse Pressure |
|-----------|-----|----------------|
| Case 1 | 34.9 | 70 mmHg |
| Case 2 | 22.9 | 40 mmHg |
| Case 3 | 30.4 | 50 mmHg |
| Case 4 | 22.0 | 40 mmHg |
| Case 5 | 26.4 | 50 mmHg |
