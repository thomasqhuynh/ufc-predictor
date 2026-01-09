import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_model, load_data

# Page config
st.set_page_config(page_title="Model Performance", page_icon="📊", layout="wide")

st.title("Model Performance & Information")

# Load data
model_pkg = load_model()
df = load_data()

if model_pkg is None or df is None:
    st.stop()

# Model Overview
st.markdown("### Model Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model Type", "Calibrated Random Forest")
with col2:
    st.metric("Features", len(model_pkg['features']))
with col3:
    st.metric("Total Fights", len(df))

st.markdown("---")

# Feature List
st.markdown("### Features Used (20 Total)")

st.markdown("""
The model uses the following 20 features, all calculated as differences (Fighter 1 - Fighter 2):

**Elo & Momentum (Calculated):**
- `elo_diff` - Elo rating difference (captures opponent quality over career)
- `win_streak_diff` - Current win streak difference  
- `loss_streak_diff` - Current loss streak difference

**Physical Attributes:**
- `height_diff` - Height difference (cm)
- `reach_diff` - Reach difference (cm)
- `age_diff` - Age difference
- `physical_adv` - Combined physical advantage (height + reach)

**Record & Experience:**
- `wins_diff` - Career wins difference
- `losses_diff` - Career losses difference
- `win_pct_diff` - Win percentage difference
- `exp_ratio` - Experience ratio (total fights)

**Striking Stats:**
- `slpm_diff` - Significant strikes landed per minute
- `sapm_diff` - Significant strikes absorbed per minute
- `str_acc_diff` - Striking accuracy difference
- `str_def_diff` - Striking defense difference
- `strike_diff_diff` - Net striking differential (output - input)

**Grappling Stats:**
- `td_avg_diff` - Takedowns per 15 minutes
- `td_acc_diff` - Takedown accuracy difference
- `td_def_diff` - Takedown defense difference
- `sub_avg_diff` - Submissions attempted per 15 minutes

All features are differences or ratios to maintain symmetry.
""")

st.markdown("---")

# Feature Importance
st.markdown("### Feature Importance")
st.markdown("Top features that influence fight predictions:")

model = model_pkg['model']

if hasattr(model, 'calibrated_classifiers_'):
    base_model = model.calibrated_classifiers_[0].estimator
else:
    base_model = model

if hasattr(base_model, 'feature_importances_'):
    importance_df = pd.DataFrame({
        'Feature': model_pkg['features'],
        'Importance': base_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    top_features = importance_df.head(20)
    sns.barplot(data=top_features, y='Feature', x='Importance', palette='viridis', ax=ax)
    ax.set_xlabel('Importance Score')
    ax.set_ylabel('Feature')
    ax.set_title('Feature Importance (All 20 Features)')
    plt.tight_layout()
    st.pyplot(fig)
    
    with st.expander("View Feature Importance Table"):
        st.dataframe(importance_df, use_container_width=True, hide_index=True)

st.markdown("---")

# Model Training Details
st.markdown("### Model Training Details")

st.markdown(f"""
**Dataset:**
- Total fights: {len(df):,}
- Date range: {df['date'].min().date()} to {df['date'].max().date()}
- Fighters: {len(set(df['r_name'].unique()) | set(df['b_name'].unique())):,}

**Training Approach:**
- **Temporal Split:** 80/20 train/test split by date
- **Model:** Random Forest with 200 trees, max depth 10
- **Calibration:** Isotonic regression for reliable probability estimates (5-fold CV)
- **Feature Engineering:** 20 hand-crafted features (no raw data passed to model)
- **Elo System:** Dynamic rating system (K=32, initial=1500) calculated chronologically
- **Streak Tracking:** Win/loss streaks calculated from fight history



**Note:** Elo ratings and streaks are calculated from the training data and frozen in the model. 
New fights or fighters won't update Elo unless the model is retrained.
""")

st.markdown("---")

# Data Source
st.markdown("### Data Source")

st.markdown("""
**Dataset:** [UFC Datasets 1994-2025](https://www.kaggle.com/datasets/neelagiriaditya/ufc-datasets-1994-2025)

**Description:**  
A comprehensive collection of all UFC fights from **March 11, 1994** to **July 12, 2025**, scraped from [ufcstats.com](http://ufcstats.com).

**Dataset Files:**
- `event_details.csv` - Event information (date, location, fight count, winners)
- `fight_details.csv` - Detailed fight statistics (strikes, takedowns, knockdowns, body shots)
- `fighter_details.csv` - Fighter profiles and career statistics
- `UFC.csv` - Master dataset (combination of all files above)

**Data Includes:**
- Fighter physical attributes (height, reach, age, weight class)
- Career records (wins, losses, draws)
- Striking stats (strikes/min, accuracy, defense, absorbed/min)
- Grappling stats (takedowns/15min, accuracy, defense, submissions)
- Complete fight history for chronological Elo calculation
- Fight-level details (strikes landed, takedowns, knockdowns, control time)

**Citation:**  
Dataset by Neelagiri Aditya, available on [Kaggle](https://www.kaggle.com/datasets/neelagiriaditya/ufc-datasets-1994-2025)
""")