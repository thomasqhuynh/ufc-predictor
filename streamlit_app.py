import streamlit as st

st.set_page_config(
    page_title="UFC Fight Predictor",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="expanded"
)

pages = [
    st.Page("pages/1_Fight_Predictor.py", title="Fight Predictor"),
    st.Page("pages/2_Model_Performance.py", title="Model Performance"),
    st.Page("pages/3_Fighter_Database.py", title="Fighter Database"),
]

pg = st.navigation(pages)
pg.run()
