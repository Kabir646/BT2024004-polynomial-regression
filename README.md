# Machine Learning Assignment: Polynomial Regression
**Student Roll Number:** `BT2024004`  
**Course:** Machine Learning  
**Topic:** Multivariate Polynomial Regression & Bias-Variance Tradeoff Analysis  
**GitHub Repository:** [Kabir646/BT2024004-polynomial-regression](https://github.com/Kabir646/BT2024004-polynomial-regression)

---

## 📌 Executive Summary
This repository contains the complete implementation, training, evaluation, and inference code for the **Polynomial Regression Assignment**.

Two real-world renewable energy engineering tasks are solved:

1. **Phase 1: Power Plant Steam Turbine Optimization (`var1`)**
   - **Inputs:** 6 operational parameters ($x_1, \dots, x_6$) representing percentage deviations.
   - **Target:** Net Power Score ($y$).
   - **Optimal Degree:** **Degree 4** (Polynomial OLS)
   - **10-Fold CV MSE:** `0.67795` | **CV R²:** `0.93066` | **Train R²:** `0.96288`
   - **Prediction File:** `BT2024004_pred_var1.csv`

2. **Phase 2: Subterranean Thermal Reservoir Mapping (`var2`)**
   - **Inputs:** 3D spatial coordinate offsets ($x_1$: East-West, $x_2$: North-South, $x_3$: Vertical depth).
   - **Target:** Thermal Anomaly Score ($y$).
   - **Optimal Degree:** **Degree 8** (Polynomial OLS)
   - **10-Fold CV MSE:** `0.29161` | **CV R²:** `0.99295` | **Train R²:** `0.99584`
   - **Prediction File:** `BT2024004_pred_var2.csv`

> All submitted test predictions use **unregularized Ordinary Least Squares (OLS)** polynomial regression at the above identified optimal degrees, in strict compliance with the assignment directive.

---

## 📂 Repository Structure
```text
BT2024004-polynomial-regression/
├── BT2024004/                       # Assigned Datasets
│   ├── BT2024004_train_var1.csv     # Phase 1 training set (1000 x 7)
│   ├── BT2024004_test_var1.csv      # Phase 1 test set (1000 x 6)
│   ├── BT2024004_train_var2.csv     # Phase 2 training set (1000 x 4)
│   └── BT2024004_test_var2.csv      # Phase 2 test set (1000 x 3)
├── src/                             # Modular Source Code
│   ├── __init__.py                  # Package initializer
│   ├── data_loader.py               # Dataset loading & verification
│   ├── models.py                    # Polynomial regression model (OLS, Ridge, Lasso)
│   └── evaluate.py                  # K-Fold CV, AIC/BIC, metrics & diagnostics
├── BT2024004_pred_var1.csv          # Phase 1 prediction deliverable (1000 rows)
├── BT2024004_pred_var2.csv          # Phase 2 prediction deliverable (1000 rows)
├── Report.pdf                       # Compiled 4-5 page technical report
├── train_and_predict.py             # Main end-to-end training & prediction pipeline
├── sample_submission.csv            # Reference format provided
├── requirements.txt                 # Python dependencies
├── .gitignore
└── README.md
```

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Kabir646/BT2024004-polynomial-regression.git
   cd BT2024004-polynomial-regression
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux / macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Running the Pipeline

### Train Models & Generate Test Predictions
```bash
python train_and_predict.py
```
This script:
- Loads the datasets from `BT2024004/`
- Runs 10-fold cross-validation for both phases
- Fits the optimal OLS polynomial models on 100% of training data
- Generates `BT2024004_pred_var1.csv` and `BT2024004_pred_var2.csv`
- Runs automatic sanity checks (1000 rows, header `y`, no NaNs/Infs)

**Output:**
```
Phase 1 10-Fold CV MSE = 0.67795, CV R² = 0.93066
Phase 2 10-Fold CV MSE = 0.29161, CV R² = 0.99295
[Phase 1] [OK] Format verified: 1000 rows
[Phase 2] [OK] Format verified: 1000 rows
```

---

## 📊 Results & Model Selection

### Phase 1: Steam Turbine Optimization (`var1`) — 6 features

| Degree | Features $K$ | Train MSE | CV MSE | CV $R^2$ | BIC |
|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | 7 | 8.66937 | 8.81122 | 0.119 | 2,208 — Underfitting |
| 2 | 28 | 2.84439 | 3.01960 | 0.696 | 1,239 — High bias |
| 3 | 84 | 0.77463 | 0.95859 | 0.903 | 325 — Good |
| **4** | **210** | **0.37440** | **0.67795** | **0.931** | **468 — ✅ Optimal** |
| 5 | 462 | 0.13586 | 1.30276 | 0.866 | 1,195 — Variance inflation |
| 6 | 924 | 0.02003 | 1258.98 | -129.2 | 4,461 — Catastrophic overfit |

**Rationale for Degree 4:** CV MSE is minimized at Degree 4. Moving to Degree 5 ($K=462 \approx N/2$) nearly doubles CV error due to variance explosion. Degree 6 ($K=924$) catastrophically overfits.

---

### Phase 2: Thermal Reservoir Mapping (`var2`) — 3 features

| Degree | Features $K$ | Train MSE | CV MSE | CV $R^2$ | BIC |
|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | 4 | 33.611 | 33.987 | 0.228 | 3,543 — Heavy underfitting |
| 4 | 35 | 3.395 | 3.780 | 0.911 | 1,464 — Missing gradients |
| 6 | 84 | 0.419 | 0.525 | 0.988 | -290 — Strong |
| 7 | 120 | 0.251 | 0.362 | 0.991 | -553 — Near optimal |
| **8** | **165** | **0.185** | **0.292** | **0.993** | **-549 — ✅ Optimal** |
| 9 | 220 | 0.165 | 0.314 | 0.993 | -282 — Overfit begins |
| 10 | 286 | 0.151 | 0.393 | 0.991 | +86 — BIC rising |
| 14 | 680 | 0.069 | 31.70 | 0.252 | 2,023 — Runge divergence |

**Rationale for Degree 8:** Degree 8 achieves the global CV MSE minimum (0.29161) with R²=0.99295. BIC confirms optimality (-549.27). Beyond Degree 10, Runge's phenomenon causes boundary oscillations and exploding test error.

---

## 📄 Report
The full technical report (`Report.pdf`) covers:
- Mathematical formulation of polynomial regression (OLS, Ridge, Lasso)
- Bias-variance tradeoff & Runge's phenomenon analysis
- 10-fold cross-validation results for all tested degrees
- AIC/BIC information criteria for model order selection
- Residual diagnostics (homoscedasticity, normality of errors)
- Condition number analysis of the polynomial design matrices

---

## ✅ Deliverables Verification

| Deliverable | File | Status |
|:---|:---|:---:|
| Phase 1 Predictions | `BT2024004_pred_var1.csv` | ✅ 1000 rows, header `y` |
| Phase 2 Predictions | `BT2024004_pred_var2.csv` | ✅ 1000 rows, header `y` |
| Report (PDF) | `Report.pdf` | ✅ 4-5 pages |
| Code Repository | This repository | ✅ Complete |

---

## 📜 License
Academic assignment submission — for educational evaluation only.
