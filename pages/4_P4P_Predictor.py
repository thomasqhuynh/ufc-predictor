import streamlit as st
import pandas as pd
import numpy as np
from utils import (
    load_model, load_data, get_all_fighters, 
    create_prediction_features
)

st.set_page_config(page_title="P4P Predictor", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        background-color: #D20A0A;
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        font-size: 1.1rem;
    }
    .comparison-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .standard-box {
        background: #2e3440;
        color: white;
    }
    .p4p-box {
        background: #667eea;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def predict_p4p_fight(fighter1, fighter2, model_pkg, df, physical_stats):
    """Predict outcome with normalized physical stats."""
    
    # Create features
    features, f1_stats, f2_stats = create_prediction_features(
        fighter1, fighter2, df, model_pkg['elo']
    )
    
    if features is None:
        return None
    
    # Normalize physical attributes
    features['height_diff'] = 0  # No height advantage
    features['reach_diff'] = 0   # No reach advantage
    features['physical_adv'] = 0  # No physical advantage
    features['age_diff'] = 0
    
    X = pd.DataFrame([features])[model_pkg['features']]
    
    X_imp = pd.DataFrame(
        model_pkg['imputer'].transform(X), 
        columns=model_pkg['features']
    )
    X_scaled = pd.DataFrame(
        model_pkg['scaler'].transform(X_imp), 
        columns=model_pkg['features']
    )
    
    model = model_pkg['model']
    prediction = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]
    
    winner = fighter1 if prediction == 1 else fighter2
    
    return {
        'predicted_winner': winner,
        'fighter1_prob': round(probabilities[1] * 100, 1),
        'fighter2_prob': round(probabilities[0] * 100, 1),
    }

def predict_standard_fight(fighter1, fighter2, model_pkg, df):
    """Standard prediction with actual physical stats."""
    
    features, f1_stats, f2_stats = create_prediction_features(
        fighter1, fighter2, df, model_pkg['elo']
    )
    
    if features is None:
        return None
    
    X = pd.DataFrame([features])[model_pkg['features']]
    
    X_imp = pd.DataFrame(
        model_pkg['imputer'].transform(X), 
        columns=model_pkg['features']
    )
    X_scaled = pd.DataFrame(
        model_pkg['scaler'].transform(X_imp), 
        columns=model_pkg['features']
    )
    
    model = model_pkg['model']
    prediction = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]
    
    winner = fighter1 if prediction == 1 else fighter2
    
    # Get actual physical stats
    actual_height_diff = features.get('height_diff', 0)
    actual_reach_diff = features.get('reach_diff', 0)
    actual_age_diff = features.get('age_diff', 0)
    
    return {
        'predicted_winner': winner,
        'fighter1_prob': round(probabilities[1] * 100, 1),
        'fighter2_prob': round(probabilities[0] * 100, 1),
        'height_diff': actual_height_diff,
        'reach_diff': actual_reach_diff,
        'age_diff': actual_age_diff,
    }

# Main page
st.markdown('<h1 style="text-align: center; color: #D20A0A;">Pound-for-Pound Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #888; font-size: 1.1rem;">See who wins when physical advantages are eliminated</p>', unsafe_allow_html=True)
st.markdown("---")

model_pkg = load_model()
df = load_data()

if model_pkg is None or df is None:
    st.stop()

all_fighters = get_all_fighters(df)

# Calculate median physical stats for normalization
st.sidebar.markdown("### Normalized Physical Stats")
st.sidebar.info("All fighters will have these attributes in P4P mode:")

normalized_stats = {
    'height': 180.0,  # cm
    'reach': 183.0,   # cm
    'age': 29.0       # years
}

st.sidebar.metric("Height", f"{normalized_stats['height']:.0f} cm")
st.sidebar.metric("Reach", f"{normalized_stats['reach']:.0f} cm")
st.sidebar.metric("Age", f"{int(normalized_stats['age'])} years")

# Fighter selection
st.markdown("### Select Fighters")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Fighter 1")
    fighter1 = st.selectbox("Select Fighter", all_fighters, key="f1")

with col2:
    st.subheader("Fighter 2")
    fighter2 = st.selectbox("Select Fighter", all_fighters, key="f2")

# Predict button
if st.button("Compare Predictions", type="primary"):
    if fighter1 == fighter2:
        st.error("Please select two different fighters")
    else:
        with st.spinner("Analyzing fight data..."):
            standard_result = predict_standard_fight(fighter1, fighter2, model_pkg, df)
            p4p_result = predict_p4p_fight(fighter1, fighter2, model_pkg, df, normalized_stats)
        
        if standard_result is None or p4p_result is None:
            st.error("Could not find stats for one or both fighters")
        else:
            st.markdown("---")
            st.markdown("## Prediction Comparison")
            
            # Side by side comparison
            comp_col1, comp_col2 = st.columns(2)
            
            with comp_col1:
                st.markdown(f"""
                <div class='comparison-box standard-box'>
                    <h3 style='text-align: center; margin-top: 0;'>Standard Prediction</h3>
                    <h2 style='text-align: center; font-size: 2.5rem; margin: 1rem 0;'>{standard_result['predicted_winner']}</h2>
                    <p style='text-align: center; font-size: 1.3rem; margin: 0;'>{max(standard_result['fighter1_prob'], standard_result['fighter2_prob'])}% Confidence</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("**Physical Differences:**")
                st.write(f"• Height diff: {standard_result['height_diff']:.1f} cm")
                st.write(f"• Reach diff: {standard_result['reach_diff']:.1f} cm")
                st.write(f"• Age diff: {standard_result['age_diff']:.1f} years")
            
            with comp_col2:
                st.markdown(f"""
                <div class='comparison-box p4p-box'>
                    <h3 style='text-align: center; margin-top: 0;'>P4P Prediction</h3>
                    <h2 style='text-align: center; font-size: 2.5rem; margin: 1rem 0;'>{p4p_result['predicted_winner']}</h2>
                    <p style='text-align: center; font-size: 1.3rem; margin: 0;'>{max(p4p_result['fighter1_prob'], p4p_result['fighter2_prob'])}% Confidence</p>
                </div>
                """, unsafe_allow_html=True)

