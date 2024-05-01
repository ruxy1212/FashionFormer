from streamlit_extras.stylable_container import stylable_container
import streamlit as st
def ch1(text, weight, key=None):
    st.markdown(f'<span style="padding: 0em !important; font-size: 2em; font-weight: {weight};">{text}</span>', unsafe_allow_html=True)
    # with stylable_container(key=key, css_styles=[
    # f"""
    # h2 {{
    #     padding: 0em !important;
    # }}
    # """,
    # ]):
    #     st.header("Dashboard")

def ch2(text, weight, key=None):
    st.markdown(f'<span style="padding: 0em !important; font-size: 1.5em; font-weight: {weight};">{text}</span>', unsafe_allow_html=True)
def ch3(text, weight, key=None):
    st.markdown(f'<span style="padding: 0em !important; font-size: 1.27em; font-weight: {weight};">{text}</span>', unsafe_allow_html=True)
def ch4(text, weight, key=None):
    st.markdown(f'<span style="padding: 0em !important; font-size: 1em; font-weight: {weight};">{text}</span>', unsafe_allow_html=True)