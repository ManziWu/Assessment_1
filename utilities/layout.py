import streamlit as st

def page_config():
    st.set_page_config(
        page_title="LegalAI",
        page_icon=":koala:",
        layout="wide",
        initial_sidebar_state="auto"  # 根据屏幕大小自动展开或折叠侧边栏
    )
