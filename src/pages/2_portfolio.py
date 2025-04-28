import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

import utils

st.title("Portfolio Analysis")

# Model selection
model_type = st.radio(
    "Select Model",
    options=["main", "extended"],
    format_func=lambda x: "Main Model" if x == "main" else "Extended Model",
    help="Choose which model to use for analysis",
)

# Button to start analysis
if st.button("Analyze All Clients"):
    start_time = time.time()
    with st.spinner("Analyzing all clients..."):
        # Get analysis results
        results = utils.analyze_all_probabilities(model_type)
        execution_time = time.time() - start_time

        if not results:
            st.error("No results available")
        else:
            # Create DataFrame for better visualization
            df = pd.DataFrame(results)

            # Display statistics
            st.subheader("Analysis Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Clients", len(df))
            with col2:
                st.metric("Average Probability", f"{df['probability'].mean():.2%}")
            with col3:
                st.metric("Median Probability", f"{df['probability'].median():.2%}")
            with col4:
                st.metric("Execution Time", f"{execution_time:.2f} seconds")

            # Display distribution
            st.subheader("Probability Distribution")
            fig, ax = plt.subplots()
            sns.histplot(data=df, x="probability", bins=20, ax=ax)
            ax.set_xlabel("Probability")
            ax.set_ylabel("Number of Clients")
            st.pyplot(fig)

            # Display top and bottom clients
            st.subheader("Client Analysis")
            col1, col2 = st.columns(2)

            with col1:
                st.write("Top 10 Clients")
                st.dataframe(
                    df.nlargest(10, "probability").style.format(
                        {"probability": "{:.2%}"}
                    )
                )

            with col2:
                st.write("Bottom 10 Clients")
                st.dataframe(
                    df.nsmallest(10, "probability").style.format(
                        {"probability": "{:.2%}"}
                    )
                )

            # Download button for full results
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Full Analysis",
                data=csv,
                file_name=f"client_analysis_{model_type}.csv",
                mime="text/csv",
            )
