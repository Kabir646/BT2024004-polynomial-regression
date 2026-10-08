"""
Generate publication-quality figures for the LaTeX Report.
Roll Number: BT2024004
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

os.makedirs("figures", exist_ok=True)

# Load data
df_tr1 = pd.read_csv(os.path.join("BT2024004", "BT2024004_train_var1.csv"))
df_te1 = pd.read_csv(os.path.join("BT2024004", "BT2024004_test_var1.csv"))
df_tr2 = pd.read_csv(os.path.join("BT2024004", "BT2024004_train_var2.csv"))
df_te2 = pd.read_csv(os.path.join("BT2024004", "BT2024004_test_var2.csv"))

X1_tr = df_tr1[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']].values
y1_tr = df_tr1['y'].values
X2_tr = df_tr2[['x1', 'x2', 'x3']].values
y2_tr = df_tr2['y'].values

kf = KFold(n_splits=10, shuffle=True, random_state=42)

# ==============================================================================
# FIGURE 1: Phase 1 Model Selection (var1)
# ==============================================================================
print("Generating Figure 1: Phase 1 Model Selection...")
degrees_v1 = list(range(1, 6))
train_mse_v1, cv_mse_v1 = [], []
train_r2_v1, cv_r2_v1 = [], []

for d in degrees_v1:
    poly = PolynomialFeatures(degree=d)
    Xp = poly.fit_transform(X1_tr)
    lr = LinearRegression()
    
    cv_mse = -cross_val_score(lr, Xp, y1_tr, cv=kf, scoring='neg_mean_squared_error').mean()
    cv_r2 = cross_val_score(lr, Xp, y1_tr, cv=kf, scoring='r2').mean()
    lr.fit(Xp, y1_tr)
    tr_mse = mean_squared_error(y1_tr, lr.predict(Xp))
    tr_r2 = r2_score(y1_tr, lr.predict(Xp))
    
    train_mse_v1.append(tr_mse)
    cv_mse_v1.append(cv_mse)
    train_r2_v1.append(tr_r2)
    cv_r2_v1.append(cv_r2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

ax1.plot(degrees_v1, train_mse_v1, 'o-', color='#1f77b4', linewidth=2, label='Train MSE')
ax1.plot(degrees_v1, cv_mse_v1, 's--', color='#d62728', linewidth=2, label='10-Fold CV MSE')
ax1.axvline(x=4, color='#2ca02c', linestyle=':', linewidth=2, label='Selected Degree (d=4)')
ax1.set_xlabel('Polynomial Degree')
ax1.set_ylabel('Mean Squared Error (MSE)')
ax1.set_title('(a) MSE vs Polynomial Degree (Phase 1)', fontweight='bold')
ax1.set_xticks(degrees_v1)
ax1.set_yscale('log')
ax1.legend(frameon=True)
ax1.grid(True, linestyle='--', alpha=0.6)

ax2.plot(degrees_v1, train_r2_v1, 'o-', color='#1f77b4', linewidth=2, label='Train $R^2$')
ax2.plot(degrees_v1, cv_r2_v1, 's--', color='#d62728', linewidth=2, label='10-Fold CV $R^2$')
ax2.axvline(x=4, color='#2ca02c', linestyle=':', linewidth=2, label='Selected Degree (d=4)')
ax2.set_xlabel('Polynomial Degree')
ax2.set_ylabel('$R^2$ Score')
ax2.set_title('(b) $R^2$ Score vs Polynomial Degree (Phase 1)', fontweight='bold')
ax2.set_xticks(degrees_v1)
ax2.set_ylim([0, 1.05])
ax2.legend(frameon=True, loc='lower right')
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("figures/fig1_var1_model_selection.png", bbox_inches='tight')
plt.close()

# ==============================================================================
# FIGURE 2: Phase 2 Model Selection (var2)
# ==============================================================================
print("Generating Figure 2: Phase 2 Model Selection...")
degrees_v2 = list(range(1, 13))
train_mse_v2, cv_mse_v2 = [], []
train_r2_v2, cv_r2_v2 = [], []

for d in degrees_v2:
    poly = PolynomialFeatures(degree=d)
    Xp = poly.fit_transform(X2_tr)
    lr = LinearRegression()
    cv_mse = -cross_val_score(lr, Xp, y2_tr, cv=kf, scoring='neg_mean_squared_error').mean()
    cv_r2 = cross_val_score(lr, Xp, y2_tr, cv=kf, scoring='r2').mean()
    lr.fit(Xp, y2_tr)
    tr_mse = mean_squared_error(y2_tr, lr.predict(Xp))
    tr_r2 = r2_score(y2_tr, lr.predict(Xp))
    
    train_mse_v2.append(tr_mse)
    cv_mse_v2.append(cv_mse)
    train_r2_v2.append(tr_r2)
    cv_r2_v2.append(cv_r2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

ax1.plot(degrees_v2, train_mse_v2, 'o-', color='#1f77b4', linewidth=2, label='Train MSE')
ax1.plot(degrees_v2, cv_mse_v2, 's--', color='#d62728', linewidth=2, label='10-Fold CV MSE')
ax1.axvline(x=8, color='#2ca02c', linestyle=':', linewidth=2, label='Selected Degree (d=8)')
ax1.set_xlabel('Polynomial Degree')
ax1.set_ylabel('Mean Squared Error (MSE)')
ax1.set_title('(a) MSE vs Polynomial Degree (Phase 2)', fontweight='bold')
ax1.set_xticks(degrees_v2)
ax1.set_yscale('log')
ax1.legend(frameon=True)
ax1.grid(True, linestyle='--', alpha=0.6)

# AIC & BIC curve
n2 = len(y2_tr)
aic_list, bic_list = [], []
for d in degrees_v2:
    poly = PolynomialFeatures(degree=d)
    Xp = poly.fit_transform(X2_tr)
    k = Xp.shape[1]
    lr = LinearRegression().fit(Xp, y2_tr)
    rss = np.sum((y2_tr - lr.predict(Xp))**2)
    mse = rss / n2
    aic = n2 * np.log(mse) + 2 * k
    bic = n2 * np.log(mse) + k * np.log(n2)
    aic_list.append(aic)
    bic_list.append(bic)

ax2.plot(degrees_v2, aic_list, '^-', color='#9467bd', linewidth=2, label='AIC')
ax2.plot(degrees_v2, bic_list, 'v--', color='#ff7f0e', linewidth=2, label='BIC')
ax2.axvline(x=8, color='#2ca02c', linestyle=':', linewidth=2, label='Selected Degree (d=8)')
ax2.set_xlabel('Polynomial Degree')
ax2.set_ylabel('Criterion Value')
ax2.set_title('(b) Information Criteria (AIC / BIC) vs Degree', fontweight='bold')
ax2.set_xticks(degrees_v2)
ax2.legend(frameon=True)
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("figures/fig2_var2_model_selection.png", bbox_inches='tight')
plt.close()

# ==============================================================================
# FIGURE 3: Residual Diagnostics & Error Distributions
# ==============================================================================
print("Generating Figure 3: Residual Diagnostics...")
poly1_best = PolynomialFeatures(degree=4)
X1_p4 = poly1_best.fit_transform(X1_tr)
lr1_best = LinearRegression().fit(X1_p4, y1_tr)
y1_fit = lr1_best.predict(X1_p4)
res1 = y1_tr - y1_fit

poly2_best = PolynomialFeatures(degree=8)
X2_p8 = poly2_best.fit_transform(X2_tr)
lr2_best = LinearRegression().fit(X2_p8, y2_tr)
y2_fit = lr2_best.predict(X2_p8)
res2 = y2_tr - y2_fit

fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), dpi=300)

# (a) Phase 1 Residuals vs Fitted
axes[0, 0].scatter(y1_fit, res1, alpha=0.5, color='#1f77b4', edgecolors='none', s=25)
axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=1.5)
axes[0, 0].set_xlabel('Fitted Values $\hat{y}$')
axes[0, 0].set_ylabel('Residuals ($y - \hat{y}$)')
axes[0, 0].set_title('(a) Phase 1 Residuals vs. Fitted Values ($d=4$)', fontweight='bold')
axes[0, 0].grid(True, linestyle='--', alpha=0.6)

# (b) Phase 1 Residual Distribution
axes[0, 1].hist(res1, bins=30, density=True, color='#1f77b4', alpha=0.7, edgecolor='black')
# overlay normal pdf
mu1, sigma1 = np.mean(res1), np.std(res1)
x_vals1 = np.linspace(mu1 - 3.5*sigma1, mu1 + 3.5*sigma1, 100)
pdf1 = (1/(sigma1 * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x_vals1 - mu1)/sigma1)**2)
axes[0, 1].plot(x_vals1, pdf1, 'r-', linewidth=2, label=f'$\mathcal{{N}}(0, {sigma1:.2f}^2)$')
axes[0, 1].set_xlabel('Residual Value')
axes[0, 1].set_ylabel('Density')
axes[0, 1].set_title('(b) Phase 1 Residual Histogram & Normal Fit', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, linestyle='--', alpha=0.6)

# (c) Phase 2 Residuals vs Fitted
axes[1, 0].scatter(y2_fit, res2, alpha=0.5, color='#2ca02c', edgecolors='none', s=25)
axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=1.5)
axes[1, 0].set_xlabel('Fitted Values $\hat{y}$')
axes[1, 0].set_ylabel('Residuals ($y - \hat{y}$)')
axes[1, 0].set_title('(c) Phase 2 Residuals vs. Fitted Values ($d=8$)', fontweight='bold')
axes[1, 0].grid(True, linestyle='--', alpha=0.6)

# (d) Phase 2 Residual Distribution
axes[1, 1].hist(res2, bins=30, density=True, color='#2ca02c', alpha=0.7, edgecolor='black')
mu2, sigma2 = np.mean(res2), np.std(res2)
x_vals2 = np.linspace(mu2 - 3.5*sigma2, mu2 + 3.5*sigma2, 100)
pdf2 = (1/(sigma2 * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x_vals2 - mu2)/sigma2)**2)
axes[1, 1].plot(x_vals2, pdf2, 'r-', linewidth=2, label=f'$\mathcal{{N}}(0, {sigma2:.2f}^2)$')
axes[1, 1].set_xlabel('Residual Value')
axes[1, 1].set_ylabel('Density')
axes[1, 1].set_title('(d) Phase 2 Residual Histogram & Normal Fit', fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("figures/fig3_residuals_diagnostics.png", bbox_inches='tight')
plt.close()

# ==============================================================================
# FIGURE 4: Regularization Techniques Comparison
# ==============================================================================
print("Generating Figure 4: Regularization Comparison...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

# Phase 1: OLS vs Ridge vs Lasso for degrees 2, 3, 4, 5
degrees_comp1 = [2, 3, 4, 5]
ols_cv_1 = [3.0196, 0.9586, 0.6780, 1.3028]
ridge_cv_1 = [3.0183, 0.9564, 0.6527, 0.4334]
lasso_cv_1 = [3.0142, 0.9363, 0.5694, 0.2987]

bar_width = 0.25
x_idx = np.arange(len(degrees_comp1))

ax1.bar(x_idx - bar_width, ols_cv_1, width=bar_width, label='OLS', color='#1f77b4', alpha=0.85)
ax1.bar(x_idx, ridge_cv_1, width=bar_width, label='Ridge (L2)', color='#ff7f0e', alpha=0.85)
ax1.bar(x_idx + bar_width, lasso_cv_1, width=bar_width, label='Lasso (L1)', color='#2ca02c', alpha=0.85)
ax1.set_xlabel('Polynomial Degree')
ax1.set_ylabel('10-Fold CV MSE')
ax1.set_title('(a) Phase 1 CV MSE across Regularization Types', fontweight='bold')
ax1.set_xticks(x_idx)
ax1.set_xticklabels([f"d={d}" for d in degrees_comp1])
ax1.legend(frameon=True)
ax1.grid(True, linestyle='--', alpha=0.6)

# Phase 2: OLS vs Ridge for degrees 6, 7, 8, 9, 10
degrees_comp2 = [6, 7, 8, 9, 10]
ols_cv_2 = [0.5254, 0.3619, 0.2916, 0.3141, 0.3930]
ridge_cv_2 = [0.5250, 0.3548, 0.2779, 0.2758, 0.2638]
x_idx2 = np.arange(len(degrees_comp2))
bar_width2 = 0.35

ax2.bar(x_idx2 - bar_width2/2, ols_cv_2, width=bar_width2, label='OLS', color='#1f77b4', alpha=0.85)
ax2.bar(x_idx2 + bar_width2/2, ridge_cv_2, width=bar_width2, label='Ridge (L2)', color='#ff7f0e', alpha=0.85)
ax2.set_xlabel('Polynomial Degree')
ax2.set_ylabel('10-Fold CV MSE')
ax2.set_title('(b) Phase 2 CV MSE: OLS vs Ridge', fontweight='bold')
ax2.set_xticks(x_idx2)
ax2.set_xticklabels([f"d={d}" for d in degrees_comp2])
ax2.legend(frameon=True)
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("figures/fig4_regularization_comparison.png", bbox_inches='tight')
plt.close()

# ==============================================================================
# FIGURE 5: Subterranean Spatial Heatmaps (Geological slices at depth offsets)
# ==============================================================================
print("Generating Figure 5: Spatial Heatmaps...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)
depth_slices = [-0.6, 0.0, 0.6]

grid_res = 100
gx = np.linspace(-1, 1, grid_res)
gy = np.linspace(-1, 1, grid_res)
GX, GY = np.meshgrid(gx, gy)

for i, z in enumerate(depth_slices):
    GZ = np.full_like(GX, z)
    pts = np.column_stack([GX.ravel(), GY.ravel(), GZ.ravel()])
    pts_poly = poly2_best.transform(pts)
    pred_grid = lr2_best.predict(pts_poly).reshape(GX.shape)
    
    cs = axes[i].contourf(GX, GY, pred_grid, levels=30, cmap='inferno')
    cbar = fig.colorbar(cs, ax=axes[i], fraction=0.046, pad=0.04)
    cbar.set_label('Thermal Score ($y$)')
    axes[i].set_xlabel('East-West offset $x_1$ (m)')
    axes[i].set_ylabel('North-South offset $x_2$ (m)')
    axes[i].set_title(f'Depth offset $x_3 = {z:.1f}$', fontweight='bold')
    axes[i].grid(True, linestyle=':', alpha=0.5)

plt.suptitle('Subterranean Thermal Anomaly Mapping Slices ($d=8$ Model)', y=1.02, fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("figures/fig5_spatial_heatmap.png", bbox_inches='tight')
plt.close()

print("All figures generated successfully in figures/ directory!")
