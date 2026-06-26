# ══════════════════════════════════════════════════════════════
#  NASA C-MAPSS Turbofan Engine Health Dashboard
#  Run: streamlit run app.py
# ══════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from sklearn.decomposition import PCA

# ── Page config ──
st.set_page_config(
    page_title="Engine Health Monitor",
    page_icon="✈️",
    layout="wide"
)

# ══════════════════════════════════════════════════════════════
#  GLOBAL FEATURE NAME LABELS
#  Used everywhere in the dashboard for consistent naming
# ══════════════════════════════════════════════════════════════

FEATURE_NAMES = {
    'unit_id'                   : 'Engine Unit ID',
    'cycle'                     : 'Flight Cycle Number (cycle)',
    'setting_1'                 : 'Altitude (setting_1) ×1000 ft',
    'setting_2'                 : 'Mach Number (setting_2)',
    's1'                        : 'Fan Inlet Temperature (s1) °R',
    's2'                        : 'LPC Outlet Temperature (s2) °R',
    's3'                        : 'HPC Outlet Temperature (s3) °R',
    's4'                        : 'LPT Outlet Temperature (s4) °R',
    's5'                        : 'Fan Inlet Pressure (s5) psia',
    's6'                        : 'Bypass Duct Pressure (s6) psia',
    's7'                        : 'HPC Outlet Pressure (s7) psia',
    's8'                        : 'Physical Fan Speed (s8) rpm',
    's9'                        : 'Physical Core Speed (s9) rpm',
    's11'                       : 'Static Pressure at HPC Outlet (s11) psia',
    's13'                       : 'Corrected Fan Speed (s13) rpm',
    's14'                       : 'Corrected Core Speed (s14) rpm',
    's15'                       : 'Bypass Ratio (s15)',
    's17'                       : 'Bleed Enthalpy (s17)',
    'op_conditions'             : 'Operating Conditions Count',
    'dataset_FD001'             : 'Sub-dataset FD001 (1 condition, HPC fault)',
    'dataset_FD002'             : 'Sub-dataset FD002 (6 conditions, HPC fault)',
    'dataset_FD003'             : 'Sub-dataset FD003 (1 condition, HPC+Fan fault)',
    'dataset_FD004'             : 'Sub-dataset FD004 (6 conditions, HPC+Fan fault)',
    'fault_mode_HPC'            : 'Fault Mode: HPC Degradation Only',
    'fault_mode_HPC_Fan'        : 'Fault Mode: HPC + Fan Degradation',
    'rolling_s14_mean'          : 'Rolling Mean of Corrected Core Speed (s14) rpm',
    'temp_pressure_interaction' : 'Temperature × Pressure Interaction (s2 × s14)',
    'cycle_ratio'               : 'Normalised Engine Age (cycle_ratio)',
    'RUL'                       : 'Remaining Useful Life — RUL (cycles)',
    'health_stage'              : 'Health Stage',
    'health_stage_encoded'      : 'Health Stage (Encoded)',
}

def get_label(col):
    """Return human-readable label for a column name."""
    return FEATURE_NAMES.get(col, col)

def label_list(cols):
    """Return list of human-readable labels for a list of column names."""
    return [get_label(c) for c in cols]

# ══════════════════════════════════════════════════════════════
#  DATA & MODEL LOADING (cached)
# ══════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    df = pd.read_csv('data/cleaned_data.csv')
    return df

@st.cache_resource
def load_models():
    reg_model  = joblib.load('models/regression_model.pkl')
    cls_model  = joblib.load('models/classification_model.pkl')
    scaler     = joblib.load('models/scaler.pkl')
    kmeans     = joblib.load('models/kmeans_model.pkl')
    pca        = joblib.load('models/pca_model.pkl')
    with open('models/feature_columns.json') as f:
        feature_cols = json.load(f)
    with open('models/train_stats.json') as f:
        train_stats = json.load(f)
    return reg_model, cls_model, scaler, kmeans, pca, feature_cols, train_stats

df = load_data()
reg_model, cls_model, scaler, kmeans, pca, feature_cols, train_stats = load_models()

# ── Sidebar branding ──
st.sidebar.title("✈️ Engine Health Monitor")
st.sidebar.markdown("**Dataset:** NASA C-MAPSS Turbofan")
st.sidebar.markdown("**Course:** MCT-341L Data Science")
st.sidebar.markdown("---")
st.sidebar.markdown("**Sensor Guide**")
st.sidebar.markdown("- **s1–s4**: Temperatures (°R)")
st.sidebar.markdown("- **s5–s7, s11**: Pressures (psia)")
st.sidebar.markdown("- **s8, s9, s13, s14**: Speeds (rpm)")
st.sidebar.markdown("- **s15, s17**: Bypass Ratio, Bleed Enthalpy")
st.sidebar.markdown("---")
st.sidebar.markdown("Navigate using the tabs above.")

# ── Main title ──
st.title("✈️ NASA Turbofan Engine Health Monitoring Dashboard")
st.markdown("An end-to-end data science pipeline for predictive maintenance "
            "of aircraft jet engines using the NASA C-MAPSS dataset.")
st.markdown("---")

# ══════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════

tab_a, tab_b, tab_c, tab_d = st.tabs([
    "📊 Tab A — Dataset Overview",
    "🔍 Tab B — Exploratory Analysis",
    "🤖 Tab C — Model Prediction",
    "🔵 Tab D — Cluster Explorer"
])

# ══════════════════════════════════════════════════════════════
#  TAB A — DATASET OVERVIEW
# ══════════════════════════════════════════════════════════════

with tab_a:
    st.header("📊 Dataset Overview")
    st.caption("Explore the cleaned NASA C-MAPSS dataset. "
               "Use the column selector to examine individual feature distributions.")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows",     f"{len(df):,}")
    col2.metric("Total Features", f"{df.shape[1]}")
    col3.metric("Missing Values", f"{df.isnull().sum().sum()}")
    col4.metric("Engine Units",
                f"{int(df['unit_id'].nunique()) if 'unit_id' in df.columns else 'N/A'}")

    st.markdown("---")

    # Dataframe viewer
    st.subheader("Cleaned Dataset")
    st.dataframe(df, use_container_width=True, height=300)
    st.caption("Sorted and filtered interactively. "
               "All 8 redundant/constant sensors removed. Zero missing values.")

    st.markdown("---")

    # Column histogram selector — show full names in dropdown
    st.subheader("Feature Distribution Explorer")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Build label→column mapping for selectbox
    col_label_map = {get_label(c): c for c in numeric_cols}
    col_labels     = list(col_label_map.keys())

    default_label = get_label('RUL') if 'RUL' in numeric_cols else col_labels[0]
    selected_label = st.selectbox(
        "Select a feature to view its distribution:",
        options=col_labels,
        index=col_labels.index(default_label)
    )
    selected_col = col_label_map[selected_label]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(df[selected_col].dropna(), bins=50,
            color='steelblue', edgecolor='white', linewidth=0.3)
    ax.axvline(df[selected_col].mean(), color='red',
               linestyle='--', linewidth=2, label='Mean')
    ax.axvline(df[selected_col].median(), color='orange',
               linestyle='--', linewidth=2, label='Median')
    ax.set_xlabel(get_label(selected_col), fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Distribution of {get_label(selected_col)}', fontsize=13)
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.caption(f"Mean = {df[selected_col].mean():.2f} | "
               f"Median = {df[selected_col].median():.2f} | "
               f"Std = {df[selected_col].std():.2f}")

    st.markdown("---")

    # Correlation heatmap — rename axes with full labels
    st.subheader("Correlation Heatmap")
    sensor_cols_heat = [c for c in numeric_cols
                        if c not in ['unit_id', 'health_stage_encoded']][:15]
    corr = df[sensor_cols_heat].corr()

    # Rename rows/cols with full labels
    corr.index   = label_list(corr.index.tolist())
    corr.columns = label_list(corr.columns.tolist())

    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig2, ax2 = plt.subplots(figsize=(16, 12))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                cmap='coolwarm', center=0, linewidths=0.3,
                ax=ax2, annot_kws={'size': 6})
    ax2.set_title('Pearson Correlation Matrix — Full Feature Names',
                  fontsize=13, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()
    st.caption("Red = strong positive correlation. "
               "Blue = strong negative correlation. "
               "Upper triangle masked for clarity.")


# ══════════════════════════════════════════════════════════════
#  TAB B — EXPLORATORY ANALYSIS
# ══════════════════════════════════════════════════════════════

with tab_b:
    st.header("🔍 Exploratory Analysis")
    st.caption("Interactively explore relationships between features. "
               "Use the controls below to customise each plot.")

    numeric_cols_b = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols_b     = ['health_stage'] if 'health_stage' in df.columns else []

    # Build label→column map for Tab B
    col_label_map_b = {get_label(c): c for c in numeric_cols_b}
    col_labels_b    = list(col_label_map_b.keys())

    col_left, col_right = st.columns(2)

    with col_left:
        x_label = st.selectbox(
            "X-axis feature:",
            options=col_labels_b,
            index=col_labels_b.index(get_label('cycle'))
            if get_label('cycle') in col_labels_b else 0
        )
    with col_right:
        y_label = st.selectbox(
            "Y-axis feature:",
            options=col_labels_b,
            index=col_labels_b.index(get_label('RUL'))
            if get_label('RUL') in col_labels_b else 1
        )

    x_feat = col_label_map_b[x_label]
    y_feat = col_label_map_b[y_label]

    show_trend = st.checkbox("Show trend line", value=True)

    colour_by = st.selectbox(
        "Colour points by:",
        options=['None'] + cat_cols_b,
        index=0
    )

    # Range filter — show full name in label
    filter_label = st.selectbox(
        "Filter data by feature range:",
        options=col_labels_b,
        index=col_labels_b.index(get_label('RUL'))
        if get_label('RUL') in col_labels_b else 0
    )
    filter_col = col_label_map_b[filter_label]

    f_min = float(df[filter_col].min())
    f_max = float(df[filter_col].max())
    f_range = st.slider(
        f"Select range for {get_label(filter_col)}:",
        min_value=f_min, max_value=f_max,
        value=(f_min, f_max)
    )

    # Guard: same feature selected for X and Y
    if x_feat == y_feat:
        st.warning("⚠️ Please select two different features for X and Y axes.")
    else:
        df_filtered = df[
            (df[filter_col] >= f_range[0]) &
            (df[filter_col] <= f_range[1])
        ]

        plot_sample = df_filtered.sample(
            min(5000, len(df_filtered)), random_state=42
        )

        fig3, ax3 = plt.subplots(figsize=(10, 6))

        if colour_by != 'None' and colour_by in df.columns:
            palette = {'Healthy': '#2ecc71',
                       'Degrading': '#f39c12',
                       'Critical': '#e74c3c'}
            for stage, color in palette.items():
                mask_s = plot_sample[colour_by] == stage
                ax3.scatter(plot_sample.loc[mask_s, x_feat],
                            plot_sample.loc[mask_s, y_feat],
                            c=color, alpha=0.3, s=8, label=stage)
            ax3.legend(fontsize=10)
        else:
            ax3.scatter(plot_sample[x_feat], plot_sample[y_feat],
                        alpha=0.3, s=8, color='steelblue')

        if show_trend:
            z = np.polyfit(plot_sample[x_feat].dropna(),
                           plot_sample[y_feat].dropna(), 1)
            p = np.poly1d(z)
            x_line = np.linspace(plot_sample[x_feat].min(),
                                  plot_sample[x_feat].max(), 100)
            ax3.plot(x_line, p(x_line), 'r-', linewidth=2, label='Trend')
            ax3.legend(fontsize=10)

        ax3.set_xlabel(get_label(x_feat), fontsize=11)
        ax3.set_ylabel(get_label(y_feat), fontsize=11)
        ax3.set_title(f'{get_label(x_feat)}  vs  {get_label(y_feat)}',
                      fontsize=12, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()
        st.caption(f"Showing {len(plot_sample):,} sampled points "
                   f"(filtered from {len(df_filtered):,} rows where "
                   f"{get_label(filter_col)} is between "
                   f"{f_range[0]:.1f} and {f_range[1]:.1f})")


# ══════════════════════════════════════════════════════════════
#  TAB C — MODEL PREDICTION
# ══════════════════════════════════════════════════════════════

with tab_c:
    st.header("🤖 Model Prediction")
    st.caption("Enter engine sensor readings to get real-time RUL prediction "
               "and health stage classification. Models are loaded from saved "
               "files — not retrained here.")

    st.info("ℹ️ Adjust the inputs below to match the engine's current readings, "
            "then click Predict.")

    key_input_features = ['cycle', 'setting_1', 'setting_2',
                           's2', 's3', 's4', 's11', 's14', 's15']

    input_vals = {}
    cols_row1 = st.columns(3)
    cols_row2 = st.columns(3)
    cols_row3 = st.columns(3)
    all_cols  = cols_row1 + cols_row2 + cols_row3

    for col_widget, feat in zip(all_cols, key_input_features):
        with col_widget:
            f_min_i = float(train_stats[feat]['min'])
            f_max_i = float(train_stats[feat]['max'])
            f_mean  = float(train_stats[feat]['mean'])
            input_vals[feat] = st.number_input(
                label=get_label(feat),          # ← full name from FEATURE_NAMES
                min_value=f_min_i,
                max_value=f_max_i,
                value=round(f_mean, 2),
                step=round((f_max_i - f_min_i) / 100, 3),
                format="%.3f"
            )

    st.markdown("---")

    if st.button("🔍 Predict Engine Health", type='primary'):

        input_row = {}
        for feat in feature_cols:
            if feat in input_vals:
                input_row[feat] = input_vals[feat]
            elif feat in train_stats:
                input_row[feat] = train_stats[feat]['mean']
            else:
                input_row[feat] = 0.0

        cycle   = float(input_vals.get('cycle', 100))
        s14_val = float(input_vals.get('s14', 8098))
        s2_val  = float(input_vals.get('s2', 597))

        input_row['rolling_s14_mean']          = s14_val
        input_row['temp_pressure_interaction']  = s2_val * s14_val
        input_row['cycle_ratio']                = cycle / 543.0

        input_df     = pd.DataFrame([input_row])[feature_cols]
        input_scaled = scaler.transform(input_df)

        rul_pred  = float(reg_model.predict(input_scaled)[0])
        rul_pred  = max(0, rul_pred)

        cls_probs   = cls_model.predict_proba(input_scaled)[0]
        class_names = ['Healthy', 'Degrading', 'Critical']

        if rul_pred > 80:
            pred_stage = 'Healthy'
        elif rul_pred >= 30:
            pred_stage = 'Degrading'
        else:
            pred_stage = 'Critical'

        cls_pred = class_names.index(pred_stage)

        col_res1, col_res2 = st.columns(2)

        with col_res1:
            st.subheader("📈 Regression Prediction")
            st.metric("Predicted RUL (Remaining Useful Life)",
                      f"{rul_pred:.1f} cycles",
                      help="How many flight cycles remain before engine failure")

            if pred_stage == 'Healthy':
                st.success(f"🟢 Health Stage: **{pred_stage}**\n\n"
                           f"Engine is operating normally. "
                           f"Next scheduled maintenance is sufficient.")
            elif pred_stage == 'Degrading':
                st.warning(f"🟡 Health Stage: **{pred_stage}**\n\n"
                           f"Engine performance declining. "
                           f"Schedule inspection at next opportunity.")
            else:
                st.error(f"🔴 Health Stage: **{pred_stage}**\n\n"
                         f"Immediate maintenance required! "
                         f"Engine has fewer than 30 cycles remaining.")

            # Threshold guide
            st.markdown("**Health Stage Thresholds:**")
            st.markdown("🟢 Healthy → RUL > 80 cycles")
            st.markdown("🟡 Degrading → RUL 30–80 cycles")
            st.markdown("🔴 Critical → RUL < 30 cycles")

        with col_res2:
            st.subheader("📊 Classification Probabilities")
            fig_prob, ax_prob = plt.subplots(figsize=(6, 3))
            colors_prob = ['#2ecc71', '#f39c12', '#e74c3c']
            ax_prob.barh(class_names, cls_probs,
                         color=colors_prob, edgecolor='black')
            ax_prob.set_xlim(0, 1)
            ax_prob.set_xlabel('Probability', fontsize=11)
            ax_prob.set_title('Random Forest Confidence per Health Stage',
                              fontsize=11)
            for i, p in enumerate(cls_probs):
                ax_prob.text(p + 0.01, i, f'{p:.3f}', va='center')
            plt.tight_layout()
            st.pyplot(fig_prob)
            plt.close()
            st.caption("Bar length = model confidence for each health stage. "
                       "Displayed health stage is derived from predicted RUL "
                       "for physical consistency.")


# ══════════════════════════════════════════════════════════════
#  TAB D — CLUSTER EXPLORER
# ══════════════════════════════════════════════════════════════

with tab_d:
    st.header("🔵 Cluster Explorer")
    st.caption("K-Means clustering (k=3) applied to the full dataset. "
               "Explore how unsupervised clusters relate to known health stages.")

    # Reconstruct feature matrix
    numeric_test = df.select_dtypes(include=[np.number])
    drop_cols_d  = ['RUL', 'health_stage_encoded', 'unit_id']
    X_dash       = numeric_test.drop(
        columns=[c for c in drop_cols_d if c in numeric_test.columns]
    ).copy()

    # Add engineered features
    if 'rolling_s14_mean' not in X_dash.columns:
        X_dash['rolling_s14_mean'] = X_dash['s14']

    if 'temp_pressure_interaction' not in X_dash.columns:
        X_dash['temp_pressure_interaction'] = X_dash['s2'] * X_dash['s14']

    if 'cycle_ratio' not in X_dash.columns:
        if 'RUL' in df.columns:
            X_dash['cycle_ratio'] = (
                df['cycle'] / (df['cycle'] + df['RUL'] + 1)
            ).values
        else:
            X_dash['cycle_ratio'] = X_dash['cycle'] / 543.0

    X_dash         = X_dash.reindex(columns=feature_cols, fill_value=0)
    X_dash_scaled  = scaler.transform(X_dash)
    cluster_labels = kmeans.predict(X_dash_scaled)
    X_dash_pca     = pca.transform(X_dash_scaled)

    # Cluster selector
    selected_cluster = st.selectbox(
        "Select a cluster to highlight:",
        options=['All clusters'] + [f'Cluster {i}' for i in range(3)]
    )

    # PCA scatter
    fig_d, ax_d = plt.subplots(figsize=(12, 7))
    cluster_colors_d = {0: '#3498db', 1: '#e74c3c', 2: '#2ecc71'}

    if selected_cluster == 'All clusters':
        for i in range(3):
            mask = cluster_labels == i
            ax_d.scatter(X_dash_pca[mask, 0], X_dash_pca[mask, 1],
                         c=cluster_colors_d[i], alpha=0.2, s=5,
                         label=f'Cluster {i}')
    else:
        sel_idx = int(selected_cluster.split()[-1])
        for i in range(3):
            mask = cluster_labels == i
            if i == sel_idx:
                ax_d.scatter(X_dash_pca[mask, 0], X_dash_pca[mask, 1],
                             c=cluster_colors_d[i], alpha=0.4, s=8,
                             label=f'Cluster {i} (selected)')
            else:
                ax_d.scatter(X_dash_pca[mask, 0], X_dash_pca[mask, 1],
                             c='lightgray', alpha=0.1, s=3,
                             label=f'Cluster {i} (other)')

    centroids_d = pca.transform(kmeans.cluster_centers_)
    for i in range(3):
        ax_d.scatter(centroids_d[i, 0], centroids_d[i, 1],
                     c=cluster_colors_d[i], s=300, marker='*',
                     edgecolors='black', linewidths=1.5, zorder=5)

    ax_d.set_xlabel('PC1 — Principal Component 1 (60.0% variance explained)',
                    fontsize=11)
    ax_d.set_ylabel('PC2 — Principal Component 2 (13.0% variance explained)',
                    fontsize=11)
    ax_d.set_title('K-Means Clusters (k=3) — PCA 2D Projection\n'
                   'Stars (★) = Cluster Centroids',
                   fontsize=13, fontweight='bold')
    ax_d.legend(fontsize=10, markerscale=3)
    plt.tight_layout()
    st.pyplot(fig_d)
    plt.close()
    st.caption("The repeated banana-shaped groups along PC1 correspond to the "
               "6 operating conditions (altitude × Mach × throttle combinations), "
               "which dominate sensor readings more than engine degradation does. "
               "K-Means discovered operating condition clusters rather than health "
               "stage clusters — a key finding of this analysis.")

    st.markdown("---")

    # Per-cluster profile table
    if selected_cluster != 'All clusters':
        sel_idx = int(selected_cluster.split()[-1])
        st.subheader(f"Cluster {sel_idx} — Feature Profile")

        profile_cols   = ['cycle', 'cycle_ratio', 's14', 's11', 's13']
        available_cols = [c for c in profile_cols if c in X_dash.columns]

        X_dash_df      = X_dash.reset_index(drop=True)
        sel_mask_reset = pd.Series(cluster_labels) == sel_idx

        cluster_mean = X_dash_df.loc[sel_mask_reset, available_cols].mean()
        overall_mean = X_dash_df[available_cols].mean()
        delta        = cluster_mean - overall_mean

        # Build display table with full names as index
        compare_df = pd.DataFrame({
            'Feature (code)' : available_cols,
            'Feature (name)' : [get_label(c) for c in available_cols],
            'Overall Mean'   : overall_mean.values.round(3),
            f'Cluster {sel_idx} Mean': cluster_mean.values.round(3),
            'Delta'          : delta.values.round(3)
        })
        st.dataframe(compare_df, use_container_width=True)
        st.caption("Delta = Cluster mean minus Overall mean. "
                   "Positive = this cluster is above average for that feature.")

        # Delta bar chart with full names
        fig_delta, ax_delta = plt.subplots(figsize=(12, 5))
        full_names    = [get_label(c) for c in available_cols]
        colors_delta  = ['#e74c3c' if v < 0 else '#3498db'
                         for v in delta.values]

        ax_delta.barh(full_names, delta.values,
                      color=colors_delta, edgecolor='black', linewidth=0.5)
        ax_delta.axvline(0, color='black', linewidth=1.5)
        ax_delta.set_xlabel('Delta from Overall Mean', fontsize=12)
        ax_delta.set_title(
            f'Cluster {sel_idx} — Feature Deviation from Overall Mean\n'
            f'Blue = above average | Red = below average',
            fontsize=12, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig_delta)
        plt.close()
        st.caption(f"Cluster {sel_idx} contains "
                   f"{int(sel_mask_reset.sum()):,} samples "
                   f"({sel_mask_reset.sum()/len(cluster_labels)*100:.1f}% of dataset). "
                   f"Blue bars = above dataset average. Red bars = below average.")
