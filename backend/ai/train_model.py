"""
Hospital Resource Management System — ML Training Pipeline
Models: Random Forest (surge classifier) + Gradient Boosting (resource regressors)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, mean_absolute_error, r2_score, mean_squared_error
)
from sklearn.utils.class_weight import compute_class_weight
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("  HOSPITAL RESOURCE MANAGEMENT — ML TRAINING PIPELINE")
print("=" * 60)

DATA_PATH = "/mnt/user-data/uploads/hospital_resource_dataset_10000_updated.csv"
df = pd.read_csv(DATA_PATH)
print(f"\n[1] Data loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n[2] Engineering features...")

# Parse date
df['allocation_date'] = pd.to_datetime(df['allocation_date'])
df['month']       = df['allocation_date'].dt.month
df['day_of_week'] = df['allocation_date'].dt.dayofweek  # 0=Mon
df['quarter']     = df['allocation_date'].dt.quarter

# Utilization rates (the key signals)
df['bed_utilization_rate']       = df['beds_occupied']       / df['beds_allocated']
df['staff_utilization_rate']     = df['staff_on_duty']       / df['staff_allocated']
df['equipment_utilization_rate'] = df['equipment_in_use']    / df['equipment_allocated']

# Slack / shortage features
df['bed_slack']       = df['beds_allocated']       - df['beds_occupied']
df['staff_slack']     = df['staff_allocated']       - df['staff_on_duty']
df['equipment_slack'] = df['equipment_allocated']   - df['equipment_in_use']

# Target
df['surge_status_int'] = df['surge_status'].astype(int)

print("   ✓ Utilization rates computed")
print("   ✓ Slack features computed")
print("   ✓ Date features extracted")

# ─────────────────────────────────────────────
# 3. ENCODE CATEGORICALS
# ─────────────────────────────────────────────
print("\n[3] Encoding categorical features...")

le_bed   = LabelEncoder()
le_staff = LabelEncoder()
le_equip = LabelEncoder()

df['bed_type_enc']       = le_bed.fit_transform(df['bed_type'])
df['staff_type_enc']     = le_staff.fit_transform(df['staff_type'])
df['equipment_type_enc'] = le_equip.fit_transform(df['equipment_type'])

print(f"   bed_type    : {dict(zip(le_bed.classes_, le_bed.transform(le_bed.classes_)))}")
print(f"   staff_type  : {dict(zip(le_staff.classes_, le_staff.transform(le_staff.classes_)))}")
print(f"   equip_type  : {dict(zip(le_equip.classes_, le_equip.transform(le_equip.classes_)))}")

# ─────────────────────────────────────────────
# 4. DEFINE FEATURE SETS
# ─────────────────────────────────────────────
FEATURES = [
    # Raw numeric
    'beds_allocated', 'beds_occupied',
    'staff_allocated', 'staff_on_duty',
    'equipment_allocated', 'equipment_in_use',
    # Engineered
    'bed_utilization_rate', 'staff_utilization_rate', 'equipment_utilization_rate',
    'bed_slack', 'staff_slack', 'equipment_slack',
    # Categorical (encoded)
    'bed_type_enc', 'staff_type_enc', 'equipment_type_enc',
    # Temporal
    'month', 'day_of_week', 'quarter',
]

TARGET_CLASS = 'surge_status_int'

X = df[FEATURES]
y_class = df[TARGET_CLASS]

print(f"\n[4] Feature matrix: {X.shape}  |  Target: {y_class.name}")
print(f"   Class balance — Surge: {y_class.sum():,} ({y_class.mean()*100:.1f}%)  "
      f"| Non-surge: {(~y_class.astype(bool)).sum():,}")

# ─────────────────────────────────────────────
# 5. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class
)
print(f"\n[5] Split — Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# Class weights to handle imbalance
classes = np.array([0, 1])
weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = {0: weights[0], 1: weights[1]}
print(f"   Class weights — {class_weight_dict}")

# ─────────────────────────────────────────────
# 6. TRAIN SURGE CLASSIFIER (Random Forest)
# ─────────────────────────────────────────────
print("\n[6] Training Surge Classifier (Random Forest)...")

clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

y_pred  = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]
auc     = roc_auc_score(y_test, y_proba)

print(f"\n   ROC-AUC Score : {auc:.4f}")
print(f"\n   Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Non-Surge', 'Surge']))

# Cross-validation
cv_scores = cross_val_score(clf, X, y_class, cv=5, scoring='roc_auc', n_jobs=-1)
print(f"   5-Fold CV AUC : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─────────────────────────────────────────────
# 7. TRAIN RESOURCE REGRESSORS
# ─────────────────────────────────────────────
print("\n[7] Training Resource Utilization Regressors (Gradient Boosting)...")

reg_features = [f for f in FEATURES if f not in
                ['bed_utilization_rate', 'staff_utilization_rate', 'equipment_utilization_rate']]

regressors = {}
reg_results = {}

for target_name, target_col in [
    ('Bed Utilization',       'bed_utilization_rate'),
    ('Staff Utilization',     'staff_utilization_rate'),
    ('Equipment Utilization', 'equipment_utilization_rate'),
]:
    y_reg = df.loc[X.index, target_col]
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X[reg_features], y_reg, test_size=0.2, random_state=42
    )
    reg = GradientBoostingRegressor(
        n_estimators=150, max_depth=5, learning_rate=0.1,
        subsample=0.8, random_state=42
    )
    reg.fit(Xr_train, yr_train)
    yr_pred = reg.predict(Xr_test)
    mae  = mean_absolute_error(yr_test, yr_pred)
    r2   = r2_score(yr_test, yr_pred)
    rmse = np.sqrt(mean_squared_error(yr_test, yr_pred))
    regressors[target_col] = reg
    reg_results[target_name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    print(f"   {target_name:25s} — MAE: {mae:.4f}  RMSE: {rmse:.4f}  R²: {r2:.4f}")

# ─────────────────────────────────────────────
# 8. FEATURE IMPORTANCE
# ─────────────────────────────────────────────
importances = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\n[8] Top 10 Feature Importances (Surge Classifier):")
for feat, imp in importances.head(10).items():
    bar = "█" * int(imp * 200)
    print(f"   {feat:35s} {imp:.4f}  {bar}")

# ─────────────────────────────────────────────
# 9. SAVE OUTPUTS
# ─────────────────────────────────────────────
os.makedirs('/home/claude/model_outputs', exist_ok=True)

# Save models
with open('/home/claude/model_outputs/surge_classifier.pkl', 'wb') as f:
    pickle.dump(clf, f)
with open('/home/claude/model_outputs/regressors.pkl', 'wb') as f:
    pickle.dump(regressors, f)
with open('/home/claude/model_outputs/encoders.pkl', 'wb') as f:
    pickle.dump({'bed': le_bed, 'staff': le_staff, 'equipment': le_equip}, f)

# Save feature list
with open('/home/claude/model_outputs/feature_list.pkl', 'wb') as f:
    pickle.dump({'features': FEATURES, 'reg_features': reg_features}, f)

print("\n[9] Models saved to /home/claude/model_outputs/")

# ─────────────────────────────────────────────
# 10. GENERATE PLOTS
# ─────────────────────────────────────────────
print("\n[10] Generating evaluation plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Hospital Resource ML — Model Evaluation', fontsize=16, fontweight='bold')

# --- Plot 1: Confusion Matrix ---
ax = axes[0, 0]
cm = confusion_matrix(y_test, y_pred)
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['Non-Surge', 'Surge']); ax.set_yticklabels(['Non-Surge', 'Surge'])
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix — Surge Classifier')
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=14)
plt.colorbar(im, ax=ax)

# --- Plot 2: ROC Curve ---
ax = axes[0, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
ax.plot(fpr, tpr, color='#1D9E75', lw=2, label=f'ROC (AUC = {auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve — Surge Classifier')
ax.legend(); ax.grid(alpha=0.3)

# --- Plot 3: Feature Importances ---
ax = axes[1, 0]
top_n = importances.head(10)
colors = ['#1D9E75' if 'util' in f or 'slack' in f else '#7F77DD' for f in top_n.index]
ax.barh(range(len(top_n)), top_n.values, color=colors)
ax.set_yticks(range(len(top_n)))
ax.set_yticklabels(top_n.index, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Importance Score')
ax.set_title('Top 10 Feature Importances')
patch1 = mpatches.Patch(color='#1D9E75', label='Engineered')
patch2 = mpatches.Patch(color='#7F77DD', label='Raw')
ax.legend(handles=[patch1, patch2], fontsize=8)
ax.grid(axis='x', alpha=0.3)

# --- Plot 4: Regressor R² Scores ---
ax = axes[1, 1]
names = list(reg_results.keys())
r2s   = [reg_results[n]['R2'] for n in names]
maes  = [reg_results[n]['MAE'] for n in names]
x_pos = np.arange(len(names))
bars  = ax.bar(x_pos, r2s, color=['#1D9E75', '#534AB7', '#D85A30'], alpha=0.85)
ax.set_xticks(x_pos)
ax.set_xticklabels([n.replace(' Utilization', '\nUtilization') for n in names], fontsize=9)
ax.set_ylabel('R² Score')
ax.set_title('Regressor R² Scores (Resource Prediction)')
ax.set_ylim(0, 1.05)
ax.grid(axis='y', alpha=0.3)
for bar, mae in zip(bars, maes):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'MAE={mae:.3f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('/home/claude/model_outputs/evaluation_plots.png', dpi=150, bbox_inches='tight')
plt.close()

print("   ✓ Plots saved: evaluation_plots.png")
print("\n" + "=" * 60)
print("  TRAINING COMPLETE")
print(f"  Surge Classifier AUC : {auc:.4f}")
print(f"  CV AUC (5-fold)      : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
for name, res in reg_results.items():
    print(f"  {name:25s} R²: {res['R2']:.4f}")
print("=" * 60)
