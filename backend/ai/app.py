"""
Hospital Resource Management System — Streamlit Dashboard
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ───────────────────────────────────────────
st.set_page_config(
    page_title="Hospital Resource Management",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── STYLING ───────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa; border-radius: 10px;
        padding: 1rem 1.2rem; border-left: 4px solid #1D9E75;
        margin-bottom: 0.5rem;
    }
    .metric-label { font-size: 13px; color: #666; margin: 0; }
    .metric-value { font-size: 26px; font-weight: 600; color: #1a1a1a; margin: 0; }
    .surge-badge {
        background: #FF4B4B; color: white; border-radius: 6px;
        padding: 4px 12px; font-size: 13px; font-weight: 600;
    }
    .safe-badge {
        background: #1D9E75; color: white; border-radius: 6px;
        padding: 4px 12px; font-size: 13px; font-weight: 600;
    }
    .section-title { font-size: 18px; font-weight: 600; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ─── LOAD DATA & MODELS ────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("/mnt/user-data/uploads/hospital_resource_dataset_10000_updated.csv")
    df['allocation_date'] = pd.to_datetime(df['allocation_date'])
    df['month']       = df['allocation_date'].dt.month
    df['day_of_week'] = df['allocation_date'].dt.dayofweek
    df['quarter']     = df['allocation_date'].dt.quarter
    df['bed_utilization_rate']       = df['beds_occupied']    / df['beds_allocated']
    df['staff_utilization_rate']     = df['staff_on_duty']    / df['staff_allocated']
    df['equipment_utilization_rate'] = df['equipment_in_use'] / df['equipment_allocated']
    df['bed_slack']       = df['beds_allocated']    - df['beds_occupied']
    df['staff_slack']     = df['staff_allocated']   - df['staff_on_duty']
    df['equipment_slack'] = df['equipment_allocated'] - df['equipment_in_use']
    df['surge_status_int'] = df['surge_status'].astype(int)

    le_bed   = LabelEncoder(); df['bed_type_enc']       = le_bed.fit_transform(df['bed_type'])
    le_staff = LabelEncoder(); df['staff_type_enc']     = le_staff.fit_transform(df['staff_type'])
    le_equip = LabelEncoder(); df['equipment_type_enc'] = le_equip.fit_transform(df['equipment_type'])

    return df, {'bed': le_bed, 'staff': le_staff, 'equipment': le_equip}

@st.cache_resource
def load_models():
    with open('/home/claude/model_outputs/surge_classifier.pkl', 'rb') as f:
        clf = pickle.load(f)
    with open('/home/claude/model_outputs/regressors.pkl', 'rb') as f:
        regs = pickle.load(f)
    return clf, regs

df, encoders = load_data()
clf, regs = load_models()

FEATURES = [
    'beds_allocated', 'beds_occupied', 'staff_allocated', 'staff_on_duty',
    'equipment_allocated', 'equipment_in_use',
    'bed_utilization_rate', 'staff_utilization_rate', 'equipment_utilization_rate',
    'bed_slack', 'staff_slack', 'equipment_slack',
    'bed_type_enc', 'staff_type_enc', 'equipment_type_enc',
    'month', 'day_of_week', 'quarter',
]
REG_FEATURES = [f for f in FEATURES if 'utilization_rate' not in f]

X = df[FEATURES]
y = df['surge_status_int']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
y_pred  = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

# ─── SIDEBAR ───────────────────────────────────────────────
st.sidebar.title("🏥 Hospital Resource Manager")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "📊 Dashboard Overview",
    "🔮 Surge Predictor",
    "📈 Model Performance",
    "📋 Feature Analysis",
    "🔍 Data Explorer"
])

# ══════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD OVERVIEW
# ══════════════════════════════════════════════════════════
if page == "📊 Dashboard Overview":
    st.title("🏥 Hospital Resource Management System")
    st.markdown("Real-time resource utilization tracking and surge prediction.")
    st.markdown("---")

    # KPI row
    surge_pct  = df['surge_status'].mean() * 100
    avg_bed    = df['bed_utilization_rate'].mean() * 100
    avg_staff  = df['staff_utilization_rate'].mean() * 100
    avg_equip  = df['equipment_utilization_rate'].mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Records", f"{len(df):,}")
    with c2:
        st.metric("Surge Events", f"{df['surge_status'].sum():,}", f"{surge_pct:.1f}%")
    with c3:
        st.metric("Avg Bed Utilization", f"{avg_bed:.1f}%")
    with c4:
        st.metric("Avg Staff Utilization", f"{avg_staff:.1f}%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    # Utilization by bed type
    with col1:
        st.subheader("Bed Utilization by Type")
        bed_util = df.groupby('bed_type')['bed_utilization_rate'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        colors = {'ICU': '#FF4B4B', 'Emergency': '#FF9500', 'General': '#1D9E75'}
        bars = ax.bar(bed_util.index, bed_util.values * 100,
                      color=[colors.get(b, '#999') for b in bed_util.index], alpha=0.85)
        ax.set_ylabel("Avg Utilization (%)")
        ax.set_ylim(0, 110)
        ax.axhline(80, color='red', linestyle='--', lw=1, alpha=0.5, label='80% threshold')
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{bar.get_height():.1f}%", ha='center', fontsize=10)
        ax.legend(); ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # Surge by month
    with col2:
        st.subheader("Surge Events by Month")
        monthly = df.groupby('month')['surge_status'].mean() * 100
        fig, ax = plt.subplots(figsize=(6, 3.5))
        month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        bar_colors = ['#FF4B4B' if v > 35 else '#1D9E75' for v in monthly.values]
        ax.bar([month_names[m-1] for m in monthly.index], monthly.values, color=bar_colors, alpha=0.85)
        ax.set_ylabel("Surge Rate (%)")
        ax.set_ylim(0, 60)
        ax.axhline(30, color='orange', linestyle='--', lw=1, alpha=0.7, label='Baseline 30%')
        ax.legend(); ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # Resource comparison: surge vs non-surge
    st.subheader("Resource Utilization: Surge vs Non-Surge")
    metrics = {
        'Bed Utilization':       ('bed_utilization_rate', '#1D9E75', '#FF4B4B'),
        'Staff Utilization':     ('staff_utilization_rate', '#534AB7', '#FF9500'),
        'Equipment Utilization': ('equipment_utilization_rate', '#185FA5', '#D85A30'),
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (label, (col, c_no, c_yes)) in zip(axes, metrics.items()):
        groups = df.groupby('surge_status')[col].mean() * 100
        bars = ax.bar(['Non-Surge', 'Surge'],
                      [groups.get(False, 0), groups.get(True, 0)],
                      color=[c_no, c_yes], alpha=0.85)
        ax.set_title(label, fontsize=11)
        ax.set_ylabel('%'); ax.set_ylim(0, 110)
        ax.grid(axis='y', alpha=0.3)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{bar.get_height():.1f}%", ha='center', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════
# PAGE 2 — SURGE PREDICTOR
# ══════════════════════════════════════════════════════════
elif page == "🔮 Surge Predictor":
    st.title("🔮 Surge Status Predictor")
    st.markdown("Enter current hospital resource parameters to predict surge risk.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🛏️ Bed Resources")
        bed_type      = st.selectbox("Bed Type", ['General', 'ICU', 'Emergency'])
        beds_alloc    = st.slider("Beds Allocated", 10, 200, 100)
        beds_occ      = st.slider("Beds Occupied", 0, beds_alloc, int(beds_alloc * 0.75))

    with col2:
        st.subheader("👩‍⚕️ Staff Resources")
        staff_type    = st.selectbox("Staff Type", ['Doctor', 'Nurse', 'Technician'])
        staff_alloc   = st.slider("Staff Allocated", 5, 100, 50)
        staff_duty    = st.slider("Staff on Duty", 0, staff_alloc, int(staff_alloc * 0.8))

    with col3:
        st.subheader("🔬 Equipment Resources")
        equip_type    = st.selectbox("Equipment Type", ['MRI', 'CT Scan', 'X-Ray', 'Ventilator'])
        equip_alloc   = st.slider("Equipment Allocated", 1, 60, 30)
        equip_use     = st.slider("Equipment in Use", 0, equip_alloc, int(equip_alloc * 0.75))

    st.markdown("---")
    month_sel   = st.slider("Month", 1, 12, 4)
    dow_sel     = st.selectbox("Day of Week", ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
    dow_map     = {'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,'Saturday':5,'Sunday':6}

    if st.button("🔍 Predict Surge Status", use_container_width=True):
        bed_util  = beds_occ  / beds_alloc
        staff_util = staff_duty / staff_alloc
        equip_util = equip_use  / equip_alloc

        input_data = pd.DataFrame([{
            'beds_allocated': beds_alloc, 'beds_occupied': beds_occ,
            'staff_allocated': staff_alloc, 'staff_on_duty': staff_duty,
            'equipment_allocated': equip_alloc, 'equipment_in_use': equip_use,
            'bed_utilization_rate': bed_util,
            'staff_utilization_rate': staff_util,
            'equipment_utilization_rate': equip_util,
            'bed_slack': beds_alloc - beds_occ,
            'staff_slack': staff_alloc - staff_duty,
            'equipment_slack': equip_alloc - equip_use,
            'bed_type_enc': encoders['bed'].transform([bed_type])[0],
            'staff_type_enc': encoders['staff'].transform([staff_type])[0],
            'equipment_type_enc': encoders['equipment'].transform([equip_type])[0],
            'month': month_sel,
            'day_of_week': dow_map[dow_sel],
            'quarter': (month_sel - 1) // 3 + 1,
        }])

        pred       = clf.predict(input_data[FEATURES])[0]
        proba_surge = clf.predict_proba(input_data[FEATURES])[0][1]

        st.markdown("---")
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            if pred == 1:
                st.markdown('<span class="surge-badge">⚠️ SURGE DETECTED</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="safe-badge">✅ NORMAL OPERATIONS</span>', unsafe_allow_html=True)
        with r2:
            st.metric("Surge Probability", f"{proba_surge*100:.1f}%")
        with r3:
            st.metric("Bed Utilization", f"{bed_util*100:.1f}%")
        with r4:
            st.metric("Staff Utilization", f"{staff_util*100:.1f}%")

        # Gauge chart
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))
        for ax, (label, val, threshold) in zip(axes, [
            ('Bed Utilization', bed_util, 0.85),
            ('Staff Utilization', staff_util, 0.90),
            ('Equipment Utilization', equip_util, 0.80),
        ]):
            color = '#FF4B4B' if val > threshold else '#1D9E75'
            ax.barh([0], [val], color=color, alpha=0.85, height=0.5)
            ax.barh([0], [1-val], left=[val], color='#e0e0e0', alpha=0.5, height=0.5)
            ax.axvline(threshold, color='orange', linestyle='--', lw=1.5)
            ax.set_xlim(0, 1); ax.set_yticks([])
            ax.set_title(f"{label}\n{val*100:.1f}%", fontsize=10)
            ax.set_xlabel("Utilization Rate")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        # Recommendations
        st.subheader("💡 Recommendations")
        recs = []
        if bed_util > 0.85:
            recs.append(f"🛏️ **Bed shortage risk** — {beds_alloc - beds_occ} beds remain. Consider activating overflow protocol.")
        if staff_util > 0.90:
            recs.append(f"👩‍⚕️ **Staff overload** — Only {staff_alloc - staff_duty} staff on standby. Activate on-call roster.")
        if equip_util > 0.80:
            recs.append(f"🔬 **Equipment pressure** — {equip_alloc - equip_use} units available. Monitor closely.")
        if not recs:
            recs.append("✅ All resources are within safe operational limits.")
        for rec in recs:
            st.markdown(f"- {rec}")

# ══════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.title("📈 Model Evaluation & Performance")
    st.markdown("---")

    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, target_names=['Non-Surge','Surge'], output_dict=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC",  f"{auc:.4f}")
    m2.metric("Accuracy", f"{report['accuracy']*100:.2f}%")
    m3.metric("Surge Recall",    f"{report['Surge']['recall']*100:.2f}%")
    m4.metric("Surge Precision", f"{report['Surge']['precision']*100:.2f}%")

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(['Non-Surge','Surge']); ax.set_yticklabels(['Non-Surge','Surge'])
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                        color='white' if cm[i,j] > cm.max()/2 else 'black', fontsize=14)
        plt.colorbar(im, ax=ax); plt.tight_layout()
        st.pyplot(fig); plt.close()

    with c2:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, color='#1D9E75', lw=2, label=f'AUC = {auc:.4f}')
        ax.plot([0,1],[0,1],'k--',lw=1)
        ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
        ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
        st.pyplot(fig); plt.close()

    # Regressor metrics
    st.markdown("---")
    st.subheader("Resource Utilization Regressors")
    reg_data = []
    for target_name, target_col in [
        ('Bed Utilization',       'bed_utilization_rate'),
        ('Staff Utilization',     'staff_utilization_rate'),
        ('Equipment Utilization', 'equipment_utilization_rate'),
    ]:
        y_reg = df.loc[X.index, target_col]
        _, Xr_test, _, yr_test = train_test_split(
            X[REG_FEATURES], y_reg, test_size=0.2, random_state=42)
        yr_pred = regs[target_col].predict(Xr_test)
        from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
        reg_data.append({
            'Model': target_name,
            'MAE':  round(mean_absolute_error(yr_test, yr_pred), 4),
            'RMSE': round(np.sqrt(mean_squared_error(yr_test, yr_pred)), 4),
            'R²':   round(r2_score(yr_test, yr_pred), 4)
        })
    st.dataframe(pd.DataFrame(reg_data), use_container_width=True, hide_index=True)

    st.image('/home/claude/model_outputs/evaluation_plots.png', use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE 4 — FEATURE ANALYSIS
# ══════════════════════════════════════════════════════════
elif page == "📋 Feature Analysis":
    st.title("📋 Feature Importance Analysis")
    st.markdown("---")

    importances = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=False)

    st.subheader("Feature Importances — Surge Classifier")
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#1D9E75' if 'util' in f or 'slack' in f
              else '#534AB7' if 'type' in f
              else '#7F77DD' for f in importances.index]
    ax.barh(range(len(importances)), importances.values, color=colors, alpha=0.85)
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels(importances.index, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score")
    ax.grid(axis='x', alpha=0.3)
    patch1 = mpatches.Patch(color='#1D9E75', label='Engineered features')
    patch2 = mpatches.Patch(color='#7F77DD', label='Raw features')
    patch3 = mpatches.Patch(color='#534AB7', label='Categorical')
    ax.legend(handles=[patch1, patch2, patch3])
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("Correlation: Utilization Rates vs Surge")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (col, label, color) in zip(axes, [
        ('bed_utilization_rate', 'Bed Utilization Rate', '#1D9E75'),
        ('staff_utilization_rate', 'Staff Utilization Rate', '#534AB7'),
        ('equipment_utilization_rate', 'Equipment Utilization Rate', '#D85A30'),
    ]):
        surge_vals    = df[df['surge_status'] == True][col]
        nonsurge_vals = df[df['surge_status'] == False][col]
        ax.hist(nonsurge_vals, bins=30, alpha=0.6, color='#1D9E75', label='Non-Surge', density=True)
        ax.hist(surge_vals,    bins=30, alpha=0.6, color='#FF4B4B', label='Surge',     density=True)
        ax.set_xlabel(label); ax.set_ylabel('Density')
        ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════
# PAGE 5 — DATA EXPLORER
# ══════════════════════════════════════════════════════════
elif page == "🔍 Data Explorer":
    st.title("🔍 Data Explorer")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        bed_filter   = st.multiselect("Bed Type",   df['bed_type'].unique().tolist(),   default=df['bed_type'].unique().tolist())
    with col2:
        staff_filter = st.multiselect("Staff Type", df['staff_type'].unique().tolist(), default=df['staff_type'].unique().tolist())
    with col3:
        surge_filter = st.multiselect("Surge Status", [True, False], default=[True, False])

    filtered = df[
        df['bed_type'].isin(bed_filter) &
        df['staff_type'].isin(staff_filter) &
        df['surge_status'].isin(surge_filter)
    ]

    st.markdown(f"**Showing {len(filtered):,} records**")
    st.dataframe(
        filtered[['allocation_date','bed_type','beds_allocated','beds_occupied',
                  'bed_utilization_rate','staff_type','staff_on_duty',
                  'equipment_type','equipment_in_use','surge_status']].round(3),
        use_container_width=True, height=400
    )

    st.download_button(
        label="⬇️ Download Filtered Data as CSV",
        data=filtered.to_csv(index=False),
        file_name="filtered_hospital_data.csv",
        mime="text/csv"
    )
