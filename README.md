# Machine Learning Assignment: Polynomial Regression
**Student Roll Number:** `BT2024004`  
**Course:** Machine Learning  
**Topic:** Multivariate Polynomial Regression & Bias-Variance Tradeoff Analysis

---

## 📌 Executive Summary
This repository contains the complete implementation, training, evaluation, inference, and documentation for the **Polynomial Regression Assignment**.

We solve two real-world renewable energy engineering tasks:
1. **Phase 1: Power Plant Steam Turbine Optimization (`var1`)**
   - **Inputs:** 6 operational parameters ($x_1, \dots, x_6$) representing percentage deviations.
   - **Target:** Net Power Score ($y$).
   - **Optimal Degree Selected:** **Degree 4** ($\text{10-Fold CV MSE} = 0.67795$, $\text{CV } R^2 = 0.93066$, $\text{Train } R^2 = 0.96288$).
   - **Output Prediction File:** `BT2024004_pred_var1.csv`

2. **Phase 2: Subterranean Thermal Reservoir Mapping (`var2`)**
   - **Inputs:** 3D spatial coordinate offsets ($x_1$: East-West, $x_2$: North-South, $x_3$: Vertical depth).
   - **Target:** Thermal Anomaly Score ($y$).
   - **Optimal Degree Selected:** **Degree 8** ($\text{10-Fold CV MSE} = 0.29161$, $\text{CV } R^2 = 0.99295$, $\text{Train } R^2 = 0.99584$).
   - **Output Prediction File:** `BT2024004_pred_var2.csv`

---

## 📂 Repository Structure
```text
ml_assignment_BT2024004/
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
├── figures/                         # High-resolution figures for report
│   ├── fig1_var1_model_selection.png
│   ├── fig2_var2_model_selection.png
│   ├── fig3_residuals_diagnostics.png
│   ├── fig4_regularization_comparison.png
│   └── fig5_spatial_heatmap.png
├── train_and_predict.py             # Main end-to-end training & prediction pipeline
├── generate_plots.py                # Script to regenerate all report figures
├── BT2024004_pred_var1.csv          # Phase 1 prediction deliverable (1000 rows)
├── BT2024004_pred_var2.csv          # Phase 2 prediction deliverable (1000 rows)
├── sample_submission.csv            # Reference format provided
├── report.tex                       # LaTeX source code for 4-5 page technical report
├── requirements.txt                 # Dependencies
├── .gitignore                       # Git ignore file
└── README.md                        # Documentation
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

## 🚀 Execution & Reproduction

### 1. Run Model Training & Generate Test Predictions
Run the unified training and prediction pipeline:
```bash
python train_and_predict.py
```
This script:
- Loads the datasets from `BT2024004/`.
- Executes 10-fold cross validation for both phases.
- Fits the optimal models on 100% of the training data.
- Generates `BT2024004_pred_var1.csv` and `BT2024004_pred_var2.csv`.
- Runs automatic sanity checks to verify the shape (1000 rows), header (`y`), and data integrity.

### 2. Generate Report Figures
To reproduce all high-resolution figures in the `figures/` directory:
```bash
python generate_plots.py
```

### 3. Compile the LaTeX Report
Compile `report.tex` into a publication-quality PDF using `pdflatex`:
```bash
pdflatex report.tex
pdflatex report.tex
```
*(Or upload `report.tex` and the `figures/` folder to Overleaf).*

---

## 📊 Experimental Results & Model Selection

### Phase 1: Steam Turbine Optimization (`var1`)
- **Dimensions:** $p = 6$ features ($x_1, \dots, x_6$).
- Feature space expands as $\binom{6 + d}{d}$.

| Degree $d$ | Features $K$ | Train MSE | 10-Fold CV MSE | CV $R^2$ | BIC | Observation |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | 7 | 8.66937 | 8.81122 | 0.11937 | 2,208.1 | Severe underfitting |
| 2 | 28 | 2.84439 | 3.01960 | 0.69639 | 1,238.8 | High bias |
| 3 | 84 | 0.77463 | 0.95859 | 0.90273 | 324.9 | Good approximation |
| **4** | **210** | **0.37440** | **0.67795** | **0.93066** | **468.2** | **Optimal OLS model (Lowest CV MSE)** |
| 5 | 462 | 0.13586 | 1.30276 | 0.86639 | 1,195.2 | Variance inflation ($p \approx N/2$) |
| 6 | 924 | 0.02003 | 1258.98 | -129.21 | 4,460.8 | Severe overfitting |

> **Rationale for Degree 4:**  
> Degree 4 achieves the lowest cross-validation error ($\text{CV MSE} = 0.67795$, $R^2 = 0.93066$). Beyond degree 4, the number of parameters increases to 462, leading to significant variance inflation on $N=1000$ training points. With Ridge regularization at $d=4$ ($\alpha=10.2$), CV MSE further improves to $0.65266$.

---

### Phase 2: Subterranean Thermal Reservoir Mapping (`var2`)
- **Dimensions:** $p = 3$ spatial coordinates ($x_1, x_2, x_3$).
- Feature space expands as $\binom{3 + d}{d}$.

| Degree $d$ | Features $K$ | Train MSE | 10-Fold CV MSE | CV $R^2$ | BIC | Observation |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | 4 | 33.61110 | 33.98738 | 0.22795 | 3,542.5 | Heavy underfitting |
| 2 | 10 | 21.37822 | 21.90553 | 0.50283 | 3,131.5 | Linear/quadratic baseline |
| 3 | 20 | 10.87558 | 11.71219 | 0.72994 | 2,524.7 | Underfitting |
| 4 | 35 | 3.39549 | 3.77970 | 0.91063 | 1,464.2 | Missing sharp gradients |
| 5 | 56 | 1.32021 | 1.58397 | 0.96265 | 664.6 | Improving |
| 6 | 84 | 0.41882 | 0.52542 | 0.98752 | -290.1 | Strong fit |
| 7 | 120 | 0.25119 | 0.36193 | 0.99132 | -552.6 | Near optimal |
| **8** | **165** | **0.18470** | **0.29161** | **0.99295** | **-549.3** | **Global optimum (Lowest CV MSE)** |
| 9 | 220 | 0.16498 | 0.31410 | 0.99250 | -282.2 | Overfitting begins |
| 10 | 286 | 0.15106 | 0.39295 | 0.99085 | 85.6 | Increasing BIC & MSE |
| 12 | 455 | 0.11147 | 1.36529 | 0.96867 | 948.3 | Oscillations emerge |
| 14 | 680 | 0.06871 | 31.70071 | 0.25167 | 2,022.6 | Runge's divergence |

> **Rationale for Degree 8:**  
> Degree 8 achieves the lowest cross-validation MSE ($0.29161$) and an exceptional $R^2$ of $0.99295$, capturing 99.3% of the subterranean spatial temperature variation. Information criteria (BIC = $-549.3$) penalizes overparameterization, confirming that higher degrees ($d \ge 10$) suffer from boundary oscillation (Runge's phenomenon).

---

## 🔍 Deliverables Verification

| Deliverable | Expected Format | Generated File | Status |
|:---|:---|:---|:---:|
| **Phase 1 Predictions** | `<ROLLNO>_pred_var1.csv` (1000 rows, header `y`) | `BT2024004_pred_var1.csv` | **Verified** |
| **Phase 2 Predictions** | `<ROLLNO>_pred_var2.csv` (1000 rows, header `y`) | `BT2024004_pred_var2.csv` | **Verified** |
| **Report** | Maximum 4-5 pages LaTeX write-up | `report.tex` | **Ready to compile** |
| **Code Repository** | Modular, reproducible codebase with Git history | Git repository | **Complete** |

---

## 📜 License
This project is submitted as an academic assignment for educational evaluation.
