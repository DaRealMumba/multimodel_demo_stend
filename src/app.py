import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Multimodel Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Application header
st.title("Multimodel Demo Application")

# Page descriptions
st.markdown("""
### Available Pages:

#### Client Profile
- Enter a client ID to view detailed client information
- Get AI-powered predictions for client communication strategy
- Receive recommendations for optimal communication channels (email/phone)
- In case it's an email strategy, generate personalized email templates based on client profile

#### Portfolio Analysis
- View overall portfolio statistics and metrics
- Analyze client distribution and segmentation
- Get insights into communication channel effectiveness
- Monitor model performance across different client segments
""")

st.write("Select a page from the sidebar menu to continue.")
