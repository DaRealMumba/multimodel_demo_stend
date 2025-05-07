import streamlit as st

from utils import (
    generate_email,
    get_available_client_ids,
    get_model_prediction,
    get_personal_info,
)

# Application header
st.title("Client Profile")


# Client ID selection
try:
    # Get list of available client IDs
    client_ids = get_available_client_ids()

    if not client_ids:
        st.error("No clients available in the database")
        client_id = None
    else:
        # Use text input with validation
        client_id_input = st.text_input(
            "Enter Client ID",
            placeholder="Enter client ID (e.g., 123)",
            help="Enter a valid client ID (total 200) from the database",
        )

        # Validate input
        if client_id_input:
            try:
                client_id = int(client_id_input)
                if client_id not in client_ids:
                    st.error(f"Client ID {client_id} not found in database")
                    client_id = None
            except ValueError:
                st.error("Please enter a valid numeric client ID")
                client_id = None
        else:
            client_id = None

except Exception as e:
    st.error(f"Error loading client list: {str(e)}")
    client_id = None

# Show remaining content only if ID is selected
if client_id is not None:
    # Show personal info
    st.header("Client Personal Information")
    personal_info = get_personal_info(client_id)
    if personal_info is not None:
        # Transpose the DataFrame for vertical display
        personal_info_transposed = personal_info.T
        st.dataframe(personal_info_transposed)
    else:
        st.error(f"Personal information for client ID {client_id} not found")

    # Button to get prediction
    if st.button("Get Prediction"):
        with st.spinner("Getting prediction..."):
            # Get prediction using data_for_model.csv
            result = get_model_prediction(client_id)

            if "error" in result:
                st.error(result["error"])
            else:
                st.success("Prediction received successfully!")

                if "Probability" in result:
                    probability = round(result["Probability"], 2)
                    st.session_state["probability"] = probability
                    st.session_state["client_data"] = (
                        personal_info  # Store personal info instead of model data
                    )
                    st.session_state["show_prediction"] = True
                else:
                    st.json(result)

    # Display prediction result (if available)
    if "show_prediction" in st.session_state and st.session_state["show_prediction"]:
        st.subheader("Prediction Result")
        probability = st.session_state["probability"]
        st.write(f"Probability: {probability}")

        # Logic for different probability levels
        if probability < 0.3:
            st.warning("⚠️ No need to contact the client")
        elif 0.3 <= probability < 0.5:
            st.info("📧 Recommended communication channel: Email")

            # Email generation section
            st.subheader("Email Generation")

            # Default prompt field
            default_prompt = """Write a marketing email for a client with the following characteristics:
- Tone: friendly and professional
- Length: medium (3-4 paragraphs)
- Content: 
  * Greeting
  * Main offer information
  * Call to action
  * Signature
The email should be persuasive but not pushy."""

            prompt = st.text_area(
                "Enter prompt for email generation",
                value=default_prompt,
                height=200,
                help="Describe what kind of email you want to generate",
            )

            if st.button("Generate Email"):
                if prompt:
                    with st.spinner("Generating email..."):
                        email_text = generate_email(prompt)
                        st.text_area("Generated Email", email_text, height=300)
                else:
                    st.warning("Please enter a prompt for email generation")
        elif probability >= 0.5:
            st.success("📞 Recommended communication channel: Phone Call")
            st.info("""
            Call recommendations:
            - Prepare a conversation script
            - Confirm convenient time for the call
            - Be ready to answer client questions
            - Document call results
            """)

    # Model information
    with st.expander("Model Information"):
        st.write("""
        - **Model**: LightGBM
        - **Output**: Probability score for communication channel
        """)
