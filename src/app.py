import streamlit as st

pages = [
    st.Page("pages/main.py", title="Main"),
    st.Page("pages/1_client_profile.py", title="Client Profile"),
    st.Page("pages/2_portfolio.py", title="Portfolio"),
]

pg = st.navigation(pages)
pg.run()
