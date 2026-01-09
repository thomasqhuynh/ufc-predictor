"""
Shared utility functions for UFC Fight Predictor
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from collections import defaultdict
import sys

# Elo System Class
class EloSystem:
    """Elo ratings + win/loss streak tracking."""
    
    def __init__(self, k=32, initial=1500):
        self.k = k
        self.initial = initial
        self.ratings = {}
        self.win_streak = defaultdict(int)
        self.loss_streak = defaultdict(int)
    
    def get(self, fighter):
        return self.ratings.get(fighter, self.initial)
    
    def expected(self, ra, rb):
        return 1 / (1 + 10 ** ((rb - ra) / 400))
    
    def update(self, winner, loser):
        ra, rb = self.get(winner), self.get(loser)
        ea = self.expected(ra, rb)
        self.ratings[winner] = ra + self.k * (1 - ea)
        self.ratings[loser] = rb + self.k * (0 - (1 - ea))
        
        # Update streaks
        self.win_streak[winner] += 1
        self.loss_streak[winner] = 0
        self.loss_streak[loser] += 1
        self.win_streak[loser] = 0
    
    def top(self, n=20):
        """Get top n fighters by Elo rating."""
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)[:n]

sys.modules['__main__'].EloSystem = EloSystem

# Load model with caching
@st.cache_resource
def load_model():
    """Load the trained model."""
    import joblib

    try:
        return joblib.load("ufc_model.joblib")
    except FileNotFoundError:
        st.error("Model file not found. Make sure 'ufc_model.joblib' is in the app directory.")
        return None


@st.cache_data
def load_data():
    """Load the UFC dataset."""
    try:
        df = pd.read_csv('UFC.csv')
        df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        st.error("Data file not found! Make sure 'UFC.csv' is in the same directory.")
        return None

def parse_height(h):
    """Parse height from format like 5'10\" to cm."""
    if pd.isna(h): 
        return np.nan
    if isinstance(h, (int, float)): 
        return h
    h = str(h).strip()
    if "'" in h:
        parts = h.replace('"', '').split("'")
        ft = int(parts[0]) if parts[0] else 0
        inch = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        return ft * 30.48 + inch * 2.54
    return float(h) if h else np.nan

def parse_reach(r):
    """Parse reach from inches to cm."""
    if pd.isna(r): 
        return np.nan
    if isinstance(r, (int, float)): 
        return r * 2.54 if r < 100 else r
    r = str(r).replace('"', '').replace("'", '').strip()
    try:
        val = float(r)
        return val * 2.54 if val < 100 else val
    except: 
        return np.nan

def calc_age(dob, fight_date):
    """Calculate age at time of fight."""
    if pd.isna(dob): 
        return np.nan
    try:
        dob = pd.to_datetime(dob)
        return (fight_date - dob).days / 365.25
    except: 
        return np.nan

def get_all_fighters(df):
    """Get sorted list of all unique fighters."""
    r_fighters = df['r_name'].unique()
    b_fighters = df['b_name'].unique()
    return sorted(list(set(list(r_fighters) + list(b_fighters))))

def get_fighter_stats(fighter_name, df):
    """Get most recent stats for a fighter."""
    # Get all fights for this fighter
    r_fights = df[df['r_name'] == fighter_name].sort_values('date', ascending=False)
    b_fights = df[df['b_name'] == fighter_name].sort_values('date', ascending=False)
    
    if r_fights.empty and b_fights.empty:
        return None, None
    
    # Get most recent fight
    if r_fights.empty:
        latest = b_fights.iloc[0]
        prefix = 'b_'
    elif b_fights.empty:
        latest = r_fights.iloc[0]
        prefix = 'r_'
    else:
        if r_fights.iloc[0]['date'] >= b_fights.iloc[0]['date']:
            latest = r_fights.iloc[0]
            prefix = 'r_'
        else:
            latest = b_fights.iloc[0]
            prefix = 'b_'
    
    # Parse physical stats from raw data
    height_raw = latest.get(f'{prefix}height')
    reach_raw = latest.get(f'{prefix}reach')
    dob = latest.get(f'{prefix}dob')
    
    height_cm = parse_height(height_raw)
    reach_cm = parse_reach(reach_raw)
    age = calc_age(dob, latest['date'])
    
    return latest, prefix, height_cm, reach_cm, age

def create_prediction_features(f1_name, f2_name, df, elo_system):
    """Create feature vector for a fight prediction."""
    # Get fighter stats
    result = get_fighter_stats(f1_name, df)
    if result[0] is None:
        return None, None, None
    f1_fight, f1_prefix, f1_height, f1_reach, f1_age = result
    
    result = get_fighter_stats(f2_name, df)
    if result[0] is None:
        return None, None, None
    f2_fight, f2_prefix, f2_height, f2_reach, f2_age = result
    
    # Extract stats
    f1 = {
        'height': f1_height,
        'reach': f1_reach,
        'age': f1_age,
        'wins': f1_fight.get(f'{f1_prefix}wins', 0),
        'losses': f1_fight.get(f'{f1_prefix}losses', 0),
        'splm': f1_fight.get(f'{f1_prefix}splm', 0),
        'sapm': f1_fight.get(f'{f1_prefix}sapm', 0),
        'str_acc': f1_fight.get(f'{f1_prefix}str_acc', 0),
        'str_def': f1_fight.get(f'{f1_prefix}str_def', 0),
        'td_avg': f1_fight.get(f'{f1_prefix}td_avg', 0),
        'td_acc': f1_fight.get(f'{f1_prefix}td_avg_acc', 0),
        'td_def': f1_fight.get(f'{f1_prefix}td_def', 0),
        'sub_avg': f1_fight.get(f'{f1_prefix}sub_avg', 0),
    }
    
    f2 = {
        'height': f2_height,
        'reach': f2_reach,
        'age': f2_age,
        'wins': f2_fight.get(f'{f2_prefix}wins', 0),
        'losses': f2_fight.get(f'{f2_prefix}losses', 0),
        'splm': f2_fight.get(f'{f2_prefix}splm', 0),
        'sapm': f2_fight.get(f'{f2_prefix}sapm', 0),
        'str_acc': f2_fight.get(f'{f2_prefix}str_acc', 0),
        'str_def': f2_fight.get(f'{f2_prefix}str_def', 0),
        'td_avg': f2_fight.get(f'{f2_prefix}td_avg', 0),
        'td_acc': f2_fight.get(f'{f2_prefix}td_avg_acc', 0),
        'td_def': f2_fight.get(f'{f2_prefix}td_def', 0),
        'sub_avg': f2_fight.get(f'{f2_prefix}sub_avg', 0),
    }
    
    # Get Elo ratings and streaks
    elo1 = elo_system.get(f1_name)
    elo2 = elo_system.get(f2_name)
    ws1 = elo_system.win_streak[f1_name]
    ws2 = elo_system.win_streak[f2_name]
    ls1 = elo_system.loss_streak[f1_name]
    ls2 = elo_system.loss_streak[f2_name]
    
    # Calculate totals
    r_total = f1['wins'] + f1['losses'] + 1
    b_total = f2['wins'] + f2['losses'] + 1
    
    # Create features (same order as training)
    features = {
        'elo_diff': elo1 - elo2,
        'height_diff': (f1['height'] if not pd.isna(f1['height']) else 0) - 
                       (f2['height'] if not pd.isna(f2['height']) else 0),
        'reach_diff': (f1['reach'] if not pd.isna(f1['reach']) else 0) - 
                      (f2['reach'] if not pd.isna(f2['reach']) else 0),
        'age_diff': f1['age'] - f2['age'],
        'wins_diff': f1['wins'] - f2['wins'],
        'losses_diff': f1['losses'] - f2['losses'],
        'win_pct_diff': f1['wins'] / r_total - f2['wins'] / b_total,
        'win_streak_diff': ws1 - ws2,
        'loss_streak_diff': ls1 - ls2,
        'slpm_diff': f1['splm'] - f2['splm'],
        'sapm_diff': f1['sapm'] - f2['sapm'],
        'str_acc_diff': f1['str_acc'] - f2['str_acc'],
        'str_def_diff': f1['str_def'] - f2['str_def'],
        'strike_diff_diff': (f1['splm'] - f1['sapm']) - (f2['splm'] - f2['sapm']),
        'td_avg_diff': f1['td_avg'] - f2['td_avg'],
        'td_acc_diff': f1['td_acc'] - f2['td_acc'],
        'td_def_diff': f1['td_def'] - f2['td_def'],
        'sub_avg_diff': f1['sub_avg'] - f2['sub_avg'],
        'physical_adv': ((f1['height'] if not pd.isna(f1['height']) else 0) - 
                        (f2['height'] if not pd.isna(f2['height']) else 0) +
                        (f1['reach'] if not pd.isna(f1['reach']) else 0) - 
                        (f2['reach'] if not pd.isna(f2['reach']) else 0)) / 2,
        'exp_ratio': r_total / b_total,
    }
    
    return features, f1, f2
