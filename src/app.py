import streamlit as st

pages = [
    st.Page("pages/main.py", title="Main"),
    st.Page("pages/client_profile.py", title="Client Profile"),
    st.Page("pages/portfolio.py", title="Portfolio"),
]

pg = st.navigation(pages)
pg.run()
