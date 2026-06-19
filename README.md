# Cirrhosis Risk Diagnosis — Deep Learning Discovery-to-Action Strategy

## Project Overview
This project builds an optimized deep learning binary classification framework using TensorFlow/Keras to screen clinical patient attributes for cirrhosis risk detection. Adhering to strict clinical constraints, this system processes physical and biochemical patient profiles to forecast risk boundaries and deliver data-driven diagnostic recommendations.

---

## Architecture and Pipeline Overview

### 1. Preprocessing & Data Cleaning
* **Missing Value Filtering:** Rows containing missing or null variables are removed to protect coordinate tracking.
* **Target Mapping:** The `Status` variable is encoded into a clean binary classification framework: `D` (Death) maps to `0`, while `C` and `CL` map to `1` (Survival/No death recorded).
* **Feature Selection:** Structural non-predictive columns (`ID`, `N_Days`, `Status`, `Drug`) are removed to avoid target leakage.
* **Categorical Handling:** Binary attributes are converted using manual dictionary indices (`F`->`1`, `M`->`0`, etc.), while multi-category groups are transformed using `pd.get_dummies()`.
* **Standardization:** All feature matrices are scaled using `StandardScaler` to stabilize gradient descent step lengths during backpropagation.

### 2. Deep Learning Neural Network Layout
* **Architecture:** A Keras `Sequential` framework consisting of two dense hidden layers containing 16 units each (utilizing `relu` activation functions), feeding into a single output neuron utilizing a `sigmoid` activation function for binary estimation.
* **Compilation:** Optimized with the `adam` gradient engine and evaluated using `binary_crossentropy` loss across exactly 10 training epochs.

---

## Strategic Clinical Summary

### Rationale for Feature Scaling
* Neural networks optimize internal weight connections via gradient descent. Wide variance differences across clinical markers (e.g., age vs. minor enzyme details) distort error surfaces. Scaling regularizes feature contributions, leading to stable model convergence.

### Clinical Error Trade-Off Evaluation
* **False Positives (Missed Diagnosis):** Predicting a patient will survive when they are actually at high risk of death is a severe clinical failure, as it deprives them of timely monitoring or critical care.
* **False Negatives (Over-Diagnosis):** Classifying a stable patient as high-risk causes psychological anxiety and triggers extra clinical testing, but preserves patient safety through close observation.
* **Recommendation:** Operational deployment should calibrate classification thresholds to prioritize high sensitivity, minimizing high-cost False Positive predictions.
