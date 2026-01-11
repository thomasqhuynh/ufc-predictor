import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from utils import (
    load_model, load_data, get_all_fighters, 
    get_fighter_stats, create_prediction_features
)

# Page config
st.set_page_config(page_title="UFC Fight Predictor", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .fighter-card {
        padding: 1.5rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #D20A0A;
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

def predict_fight(fighter1, fighter2, model_pkg, df):
    """Predict outcome of a fight."""
    
    # Create features
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
    
    # Elo and streaks
    elo1 = model_pkg['elo'].get(fighter1)
    elo2 = model_pkg['elo'].get(fighter2)
    ws1 = model_pkg['elo'].win_streak[fighter1]
    ws2 = model_pkg['elo'].win_streak[fighter2]
    ls1 = model_pkg['elo'].loss_streak[fighter1]
    ls2 = model_pkg['elo'].loss_streak[fighter2]
    
    return {
        'fighter1': fighter1,
        'fighter2': fighter2,
        'predicted_winner': winner,
        'fighter1_prob': round(probabilities[1] * 100, 1),
        'fighter2_prob': round(probabilities[0] * 100, 1),
        'fighter1_elo': round(elo1, 1),
        'fighter2_elo': round(elo2, 1),
        'fighter1_stats': f1_stats,
        'fighter2_stats': f2_stats,
        'fighter1_streak': f"W{ws1}" if ws1 > 0 else f"L{ls1}" if ls1 > 0 else "-",
        'fighter2_streak': f"W{ws2}" if ws2 > 0 else f"L{ls2}" if ls2 > 0 else "-",
        'X_scaled': X_scaled,
        'features': features
    }

def calculate_shap_values(model_pkg, X_scaled):
    """Calculate SHAP values for feature importance."""
    try:
        model = model_pkg['model']
        
        if hasattr(model, 'calibrated_classifiers_'):
            base_model = model.calibrated_classifiers_[0].estimator
        else:
            base_model = model
        
        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(X_scaled)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        return explainer, shap_values
    except Exception as e:
        st.error(f"Could not calculate SHAP values: {e}")
        return None, None

st.markdown('<h1 style="text-align: center; color: #D20A0A;">UFC Fight Predictor</h1>', unsafe_allow_html=True)
st.markdown("---")

model_pkg = load_model()
df = load_data()

if model_pkg is None or df is None:
    st.stop()

all_fighters = get_all_fighters(df)

st.markdown("### Select Fighters")

# Fighter selection
col1, col2 = st.columns(2)

with col1:
    st.subheader("Fighter 1")
    fighter1 = st.selectbox("Select Fighter", all_fighters, key="f1")

with col2:
    st.subheader("Fighter 2")
    fighter2 = st.selectbox("Select Fighter", all_fighters, key="f2")


# Predict button
if st.button("Predict Fight Outcome", type="primary"):
    if fighter1 == fighter2:
        st.error("Please select two different fighters")
    else:
        with st.spinner("Analyzing fight data..."):
            result = predict_fight(fighter1, fighter2, model_pkg, df)
        
        if result is None:
            st.error("Could not find stats for one or both fighters")
        else:
            st.markdown("---")
            st.markdown("## Prediction Result")
            
            winner_col1, winner_col2, winner_col3 = st.columns([1, 2, 1])
            with winner_col2:
                st.markdown(f"""
                <div style='text-align: center; padding: 2rem; background: #2e3440; 
                            border-radius: 15px; color: white;'>
                    <h2 style='margin: 0;'>Predicted Winner</h2>
                    <h1 style='font-size: 3rem; margin: 1rem 0;'>{result['predicted_winner']}</h1>
                    <p style='font-size: 1.5rem; margin: 0;'>{max(result['fighter1_prob'], result['fighter2_prob'])}% Confidence</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # SHAP Feature Importance
            with st.spinner("Calculating feature importance..."):
                explainer, shap_values = calculate_shap_values(model_pkg, result['X_scaled'])
                
                if explainer is not None and shap_values is not None:
                    try:
                        num_features = len(result['X_scaled'].columns)
                        
                        if len(shap_values.flatten()) == num_features * 2:
                            shap_vals = shap_values.flatten()[num_features:] 
                        else:
                            shap_vals = shap_values.flatten()
                        
                        feature_importance = pd.DataFrame({
                            'Feature': result['X_scaled'].columns,
                            'Impact': shap_vals
                        })
                        
                        feature_importance['Abs_Impact'] = feature_importance['Impact'].abs()
                        feature_importance = feature_importance.sort_values('Abs_Impact', ascending=False)
                        
                        feature_names = {
                            'elo_diff': 'Fighter Ranking (Elo)',
                            'height_diff': 'Height Advantage',
                            'reach_diff': 'Reach Advantage',
                            'age_diff': 'Age/Prime',
                            'win_pct_diff': 'Win Percentage',
                            'win_streak_diff': 'Current Momentum',
                            'loss_streak_diff': 'Loss Streak',
                            'wins_diff': 'Total Wins',
                            'losses_diff': 'Total Losses',
                            'slpm_diff': 'Striking Volume',
                            'sapm_diff': 'Strikes Absorbed',
                            'str_acc_diff': 'Strike Accuracy',
                            'str_def_diff': 'Strike Defense',
                            'td_avg_diff': 'Takedown Average',
                            'td_acc_diff': 'Takedown Accuracy',
                            'td_def_diff': 'Takedown Defense',
                            'sub_avg_diff': 'Submission Average',
                            'strike_diff_diff': 'Strike Differential',
                            'physical_adv': 'Physical Advantages',
                            'exp_ratio': 'Experience'
                        }
                        
                        categories = {
                            'Rankings & Momentum': ['elo_diff', 'win_pct_diff', 'win_streak_diff', 'loss_streak_diff', 'exp_ratio', 'wins_diff', 'losses_diff'],
                            'Striking': ['slpm_diff', 'sapm_diff', 'str_acc_diff', 'str_def_diff', 'strike_diff_diff'],
                            'Grappling': ['td_avg_diff', 'td_acc_diff', 'td_def_diff', 'sub_avg_diff'],
                            'Physical Attributes': ['height_diff', 'reach_diff', 'physical_adv', 'age_diff']
                        }
                        
                        top_factors = feature_importance[feature_importance['Impact'] > 0].head(8)
                        
                        if not top_factors.empty:
                            st.markdown("### Winning Factors")
                            st.markdown(f"**Top 3 Key Advantages for {result['predicted_winner']}:**")
                            
                            for idx, row in top_factors.head(3).iterrows():
                                feature_display = feature_names.get(row['Feature'], row['Feature'])
                                st.write(f"• {feature_display}")
                            
                            st.markdown("---")
                            st.markdown("**Detailed Breakdown of Advantages:**")
                            
                            for category, features in categories.items():
                                category_factors = top_factors[top_factors['Feature'].isin(features)]
                                if not category_factors.empty:
                                    st.markdown(f"#### {category}")
                                    for idx, row in category_factors.iterrows():
                                        feature_display = feature_names.get(row['Feature'], row['Feature'])
                                        st.write(f"  • {feature_display}")
                        else:
                            st.info("This is a very close matchup with no clear advantages.")
                    
                    except Exception as e:
                        st.error(f"Could not analyze winning factors: {e}")
            
            st.markdown("---")
            
            # Elo ratings and streaks
            st.markdown("### Elo Ratings & Streaks")
            
            elo_col1, elo_col2, elo_col3, elo_col4 = st.columns(4)
            
            with elo_col1:
                st.metric(f"{result['fighter1']} Elo", f"{result['fighter1_elo']:.0f}")
            
            with elo_col2:
                st.metric(f"{result['fighter1']} Streak", result['fighter1_streak'])
            
            with elo_col3:
                st.metric(f"{result['fighter2']} Elo", f"{result['fighter2_elo']:.0f}")
            
            with elo_col4:
                st.metric(f"{result['fighter2']} Streak", result['fighter2_streak'])
            
            st.markdown("---")
            
            # Fighter stats comparison
            st.markdown("### Fighter Stats Comparison")
            
            f1 = result['fighter1_stats']
            f2 = result['fighter2_stats']
            
            stats_data = {
                'Stat': [
                    'Record',
                    'Win %',
                    'Height (cm)',
                    'Reach (cm)',
                    'Age',
                    'Strikes/Min',
                    'Strike Accuracy',
                    'Strike Defense',
                    'TD Average',
                    'TD Accuracy',
                    'TD Defense',
                    'Sub Average'
                ],
                result['fighter1']: [
                    f"{f1['wins']}-{f1['losses']}",
                    f"{f1['wins'] / (f1['wins'] + f1['losses'] + 1) * 100:.1f}%",
                    f"{f1['height']:.1f}" if not pd.isna(f1['height']) else "N/A",
                    f"{f1['reach']:.1f}" if not pd.isna(f1['reach']) else "N/A",
                    f"{int(f1['age'])}" if not pd.isna(f1['age']) else "N/A", 
                    f"{f1['splm']:.2f}",
                    f"{f1['str_acc']:.1f}%",
                    f"{f1['str_def']:.1f}%", 
                    f"{f1['td_avg']:.2f}",
                    f"{f1['td_acc']:.1f}%",  
                    f"{f1['td_def']:.1f}%",  
                    f"{f1['sub_avg']:.2f}"
                ],
                result['fighter2']: [
                    f"{f2['wins']}-{f2['losses']}",
                    f"{f2['wins'] / (f2['wins'] + f2['losses'] + 1) * 100:.1f}%",
                    f"{f2['height']:.1f}" if not pd.isna(f2['height']) else "N/A",
                    f"{f2['reach']:.1f}" if not pd.isna(f2['reach']) else "N/A",
                    f"{int(f2['age'])}" if not pd.isna(f2['age']) else "N/A",
                    f"{f2['splm']:.2f}",
                    f"{f2['str_acc']:.1f}%",
                    f"{f2['str_def']:.1f}%",
                    f"{f2['td_avg']:.2f}",
                    f"{f2['td_acc']:.1f}%", 
                    f"{f2['td_def']:.1f}%", 
                    f"{f2['sub_avg']:.2f}"
                ],
            }
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)