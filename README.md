# Multimodel Demo Application

This is a Streamlit-based application that provides client profile analysis and communication channel recommendations.

## Features

- Client profile viewing with personal information
- Machine learning model predictions for communication channel selection
- Email generation using GPT
- Secure data handling with Streamlit secrets

## Setup

1. Clone the repository
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install uv in the virtual environment:
   ```bash
   pip install uv
   ```
4. Install project dependencies:
   ```bash
   uv pip install -e .
   ```

## Configuration

1. Create a `.streamlit` folder in the project root
2. Create `secrets.toml` in the `.streamlit` folder with the following structure:
   ```toml
   data = """
   contents of data_for_model.csv
   """

   personal_info = """
   contents of personal_info.csv
   """

   cert_file = """
   contents of your certificate
   """

   OPEN_AI_API_KEY = "your-openai-api-key"
   token_url = "your-token-url"
   main_model_url = "your-main-model-url"
   extended_model_url = "your-extended-model-url"
   ```

## Running the Application

```bash
streamlit run src/app.py
```

## Project Structure

```
.
├── .streamlit/              # Streamlit configuration
│   └── secrets.toml         # Secrets and configuration
├── src/                     # Source code
│   ├── app.py               # Main application entry point
│   ├── pages/               # Application pages
│   │   ├── 1_client_profile.py  # Client profile page
│   │   └── 2_portfolio      # Portfolio page
│   └── utils.py             # Utility functions
├── data/                    # Data directory (not in git)
├── pyproject.toml           # Project configuration and dependencies
└── README.md                # Project documentation
```

## Security

- All sensitive data is stored in Streamlit secrets
- Data files are created as temporary files at runtime
- Certificate and API keys are securely managed

## Dependencies

Dependencies are managed through `pyproject.toml` using uv. Main dependencies include:
- streamlit
- pandas
- openai
- python-dotenv
- httpx
- requests

## License

[Your License Here]