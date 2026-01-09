import streamlit as st
import pandas as pd
from utils import load_model, load_data, get_all_fighters, get_fighter_stats

st.set_page_config(page_title="Fighter Database", layout="wide")

st.title("Fighter Database")

model_pkg = load_model()
df = load_data()

if model_pkg is None or df is None:
    st.stop()

all_fighters = get_all_fighters(df)
elo_system = model_pkg['elo']

# Fighter Search
st.markdown("### Fighter Search")

search_query = st.text_input("Search for a fighter:", placeholder="Enter fighter name...")

if search_query:
    matching_fighters = [f for f in all_fighters if search_query.lower() in f.lower()]
    
    if matching_fighters:
        st.write(f"Found {len(matching_fighters)} fighter(s):")
        
        selected_fighter = st.selectbox("Select a fighter:", matching_fighters)
        
        if selected_fighter:
            st.markdown(f"### {selected_fighter}")
            
            # All fights for this fighter
            r_fights = df[df['r_name'] == selected_fighter]
            b_fights = df[df['b_name'] == selected_fighter]
            all_fights = pd.concat([r_fights, b_fights]).sort_values('date', ascending=False)
            
            # Calculate stats using winner column
            r_wins = len(r_fights[r_fights['winner'] == r_fights['r_name']])
            b_wins = len(b_fights[b_fights['winner'] == b_fights['b_name']])
            total_wins = r_wins + b_wins
            
            r_losses = len(r_fights[r_fights['winner'] != r_fights['r_name']])
            b_losses = len(b_fights[b_fights['winner'] != b_fights['b_name']])
            total_losses = r_losses + b_losses
            
            # Get latest stats
            result = get_fighter_stats(selected_fighter, df)
            if result[0] is None:
                st.error("No fights found!")
                st.stop()
            latest, prefix, height, reach, age = result
            
            # Get streak
            ws = elo_system.win_streak[selected_fighter]
            ls = elo_system.loss_streak[selected_fighter]
            streak = f"W{ws}" if ws > 0 else f"L{ls}" if ls > 0 else "-"
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Elo Rating", f"{elo_system.get(selected_fighter):.0f}")
            with col2:
                st.metric("Record", f"{total_wins}-{total_losses}")
            with col3:
                win_pct = (total_wins / (total_wins + total_losses) * 100) if (total_wins + total_losses) > 0 else 0
                st.metric("Win %", f"{win_pct:.1f}%")
            with col4:
                st.metric("Streak", streak)
            
            # Physical stats
            st.markdown("#### Physical Stats")
            phys_col1, phys_col2, phys_col3 = st.columns(3)
            
            with phys_col1:
                st.metric("Height", f"{height:.1f} cm" if not pd.isna(height) else "N/A")
            with phys_col2:
                st.metric("Reach", f"{reach:.1f} cm" if not pd.isna(reach) else "N/A")
            with phys_col3:
                st.metric("Age", f"{int(age)}" if not pd.isna(age) else "N/A")  # Round down
            
            # Fighting stats
            st.markdown("#### Fighting Stats")
            fight_col1, fight_col2, fight_col3, fight_col4 = st.columns(4)
            
            with fight_col1:
                st.metric("Strikes/Min", f"{latest.get(f'{prefix}splm', 0):.2f}")
            with fight_col2:
                st.metric("Strike Accuracy", f"{latest.get(f'{prefix}str_acc', 0):.1f}%")  # Already a percentage
            with fight_col3:
                st.metric("TD Average", f"{latest.get(f'{prefix}td_avg', 0):.2f}")
            with fight_col4:
                st.metric("TD Accuracy", f"{latest.get(f'{prefix}td_avg_acc', 0):.1f}%")  # Already a percentage
            
            # Recent fights
            st.markdown("#### Recent Fights")
            recent_fights = all_fights.head(10)
            
            recent_data = []
            for _, fight in recent_fights.iterrows():
                if fight['r_name'] == selected_fighter:
                    opponent = fight['b_name']
                    result = 'Win' if fight['winner'] == selected_fighter else 'Loss'
                else:
                    opponent = fight['r_name']
                    result = 'Win' if fight['winner'] == selected_fighter else 'Loss'
                
                recent_data.append({
                    'Date': fight['date'].date(),
                    'Opponent': opponent,
                    'Result': result
                })
            
            recent_df = pd.DataFrame(recent_data)
            st.dataframe(recent_df, use_container_width=True, hide_index=True)
    else:
        st.warning("No fighters found matching your search.")
else:
    st.info("Enter a fighter name to search")

st.markdown("---")

# Top Fighters by Elo
st.markdown("### Top Fighters by Elo Rating")

top_n = st.slider("Number of fighters to display", 10, 50, 20)
top_fighters = elo_system.top(top_n)

top_fighters_data = []
for rank, (fighter, rating) in enumerate(top_fighters, 1):
    # Get fighter stats
    r_fights = df[df['r_name'] == fighter]
    b_fights = df[df['b_name'] == fighter]
    
    # Count wins/losses using winner column
    r_wins = len(r_fights[r_fights['winner'] == r_fights['r_name']])
    b_wins = len(b_fights[b_fights['winner'] == b_fights['b_name']])
    total_wins = r_wins + b_wins
    
    r_losses = len(r_fights[r_fights['winner'] != r_fights['r_name']])
    b_losses = len(b_fights[b_fights['winner'] != b_fights['b_name']])
    total_losses = r_losses + b_losses
    
    # Get streak
    ws = elo_system.win_streak[fighter]
    ls = elo_system.loss_streak[fighter]
    streak = f"W{ws}" if ws > 0 else f"L{ls}" if ls > 0 else "-"
    
    top_fighters_data.append({
        'Rank': rank,
        'Fighter': fighter,
        'Elo Rating': f"{rating:.0f}",
        'Record': f"{total_wins}-{total_losses}",
        'Streak': streak
    })

top_df = pd.DataFrame(top_fighters_data)
st.dataframe(top_df, use_container_width=True, hide_index=True)