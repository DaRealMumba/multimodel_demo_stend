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

            # Add decile column
            df["decile"] = pd.qcut(
                df["probability"], q=10, labels=[f"Decile {i + 1}" for i in range(10)]
            )

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
            col1, col2 = st.columns(2)

            with col1:
                # Histogram
                fig1, ax1 = plt.subplots()
                sns.histplot(data=df, x="probability", bins=20, ax=ax1)
                ax1.set_xlabel("Probability")
                ax1.set_ylabel("Number of Clients")
                st.pyplot(fig1)

            with col2:
                # Decile boxplot
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                sns.boxplot(data=df, x="decile", y="probability", ax=ax2)
                ax2.set_xlabel("Decile")
                ax2.set_ylabel("Probability")
                ax2.set_title("Probability Distribution by Decile")
                plt.xticks(rotation=45)
                st.pyplot(fig2)

            # Display decile statistics
            st.subheader("Decile Statistics")
            decile_stats = (
                df.groupby("decile")["probability"]
                .agg(["mean", "median", "count"])
                .round(4)
            )
            decile_stats["mean"] = decile_stats["mean"].map("{:.2%}".format)
            decile_stats["median"] = decile_stats["median"].map("{:.2%}".format)
            st.dataframe(decile_stats)

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
