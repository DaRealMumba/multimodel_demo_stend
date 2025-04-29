import asyncio
import os
from typing import Any, Dict, List, Optional

import httpx
import openai
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

# Try to load from .env file first, then fall back to Streamlit secrets
try:
    load_dotenv()
except ImportError:
    pass

# Get configuration from environment variables or Streamlit secrets
CERT_FILE = os.getenv("cert_file") or st.secrets["cert_file"]
TOKEN_URL = os.getenv("token_url") or st.secrets["token_url"]
MAIN_MODEL_URL = os.getenv("main_model_url") or st.secrets["main_model_url"]
EXTENDED_MODEL_URL = os.getenv("extended_model_url") or st.secrets["extended_model_url"]
OPENAI_API_KEY = os.getenv("OPEN_AI_API_KEY") or st.secrets["OPEN_AI_API_KEY"]


# OpenAI setup
client = openai.OpenAI(api_key=OPENAI_API_KEY)


def initialize_data_files():
    """Initialize data files from Streamlit secrets"""
    files_to_create = {
        "data/model_data.csv": st.secrets["data"],
        "data/personal.csv": st.secrets["personal_info"],
        "cert/model_cert.pem": st.secrets["cert_file"],
    }

    # Create directories and write files
    for relative_path, content in files_to_create.items():
        full_path = os.path.join("/tmp", relative_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)

    return {
        "model_data": "/tmp/data/model_data.csv",
        "personal_info": "/tmp/data/personal.csv",
        "cert_file": "/tmp/cert/model_cert.pem",
    }


# Initialize data files and get paths
data_paths = initialize_data_files()
DATA_PATH = data_paths["model_data"]
PERSONAL_INFO_PATH = data_paths["personal_info"]
CERT_FILE = data_paths["cert_file"]


# Чтение CSV
@st.cache_data
def load_data(filepath: str, delimiter: str = ",") -> pd.DataFrame:
    """Load data from CSV file

    Args:
        filepath (str): Path to the CSV file
        delimiter (str, optional): Delimiter used in the CSV file. Defaults to ",".

    Returns:
        pd.DataFrame: Loaded data
    """
    return pd.read_csv(filepath, delimiter=delimiter)


# Load data
csv_data = load_data(DATA_PATH)
json_data = load_data(PERSONAL_INFO_PATH, delimiter=";")


@st.cache_data
def get_available_client_ids() -> List[int]:
    """
    Get list of available client IDs from CSV file.

    Returns:
        List[int]: List of client IDs (indices)
    """
    try:
        clients_df = load_data(DATA_PATH)
        return sorted(clients_df.index.tolist())
    except FileNotFoundError:
        raise Exception("Client data file not found")
    except pd.errors.EmptyDataError:
        raise Exception("Client data file is empty")
    except pd.errors.ParserError:
        raise Exception("Error parsing client data file")
    except Exception as e:
        raise Exception(f"Error getting client list: {str(e)}")


@st.cache_data
def get_personal_info(client_id: int) -> Optional[pd.DataFrame]:
    """
    Get personal information for a client from personal_info.csv.

    Args:
        client_id (int): Client ID (index)

    Returns:
        Optional[pd.DataFrame]: DataFrame with personal info or None if client not found
    """
    try:
        # Load personal info data
        personal_info_df = load_data(PERSONAL_INFO_PATH, delimiter=";")

        # Get specific client data by index
        if client_id in personal_info_df.index:
            return personal_info_df.loc[[client_id]]
        return None

    except FileNotFoundError:
        raise Exception("Personal info file not found")
    except pd.errors.EmptyDataError:
        raise Exception("Personal info file is empty")
    except pd.errors.ParserError:
        raise Exception("Error parsing personal info file")
    except Exception as e:
        raise Exception(f"Error getting personal info: {str(e)}")


@st.cache_data
def get_client_data(client_id: int) -> Optional[pd.DataFrame]:
    """
    Get client data by ID (index).

    Args:
        client_id (int): Client ID (index)

    Returns:
        Optional[pd.DataFrame]: DataFrame with client data or None if client not found
    """
    try:
        # Load data
        clients_df = load_data(DATA_PATH)

        # Get specific client data by index
        if client_id in clients_df.index:
            return clients_df.loc[[client_id]]
        return None

    except FileNotFoundError:
        raise Exception("Client data file not found")
    except pd.errors.EmptyDataError:
        raise Exception("Client data file is empty")
    except pd.errors.ParserError:
        raise Exception("Error parsing client data file")
    except Exception as e:
        raise Exception(f"Error getting client data: {str(e)}")


@st.cache_data(ttl=3600)  # Cache token for 1 hour
def get_auth_token() -> str:
    """
    Get authorization token.

    Returns:
        str: Access token
    """
    auth_data = {
        "username": "user1",
        "password": "password1",
    }

    response = requests.post(TOKEN_URL, data=auth_data, verify=CERT_FILE)

    if response.status_code == 200:
        return response.json().get("access_token")
    else:
        raise Exception(f"Failed to get access token: {response.text}")


@st.cache_data
def get_model_prediction(client_id: int, model_type: str = "main") -> Dict[str, Any]:
    """
    Get model prediction for client.

    Args:
        client_id (int): Client ID (index)
        model_type (str): Type of model to use ("main" or "extended")

    Returns:
        Dict[str, Any]: Model prediction result
    """
    try:
        # Get client data
        client_data = get_client_data(client_id)

        if client_data is None:
            return {"error": f"Client with ID {client_id} not found"}

        # Get token
        token = get_auth_token()

        # Prepare session with token
        session = requests.Session()
        session.headers.update({"Authorization": f"Bearer {token}"})

        # Optimize data processing
        row_data = client_data.iloc[0]
        row_json = {
            k: v.isoformat() if isinstance(v, pd.Timestamp) else v
            for k, v in row_data.items()
            if pd.notna(v)  # Skip NaN values
        }

        # Select model URL based on model_type
        model_url = MAIN_MODEL_URL if model_type == "main" else EXTENDED_MODEL_URL

        # Send request to model with timeout
        payload = {"row": row_json}
        response = session.post(
            model_url,
            json=payload,
            verify=CERT_FILE,
            timeout=30,  # Add timeout to prevent hanging
        )

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Model request error: {response.status_code}"}

    except requests.Timeout:
        return {"error": "Request timed out"}
    except Exception as e:
        return {"error": f"Error getting prediction: {str(e)}"}


@st.cache_data
def generate_email(prompt: str) -> str:
    """
    Generate email text using GPT.

    Args:
        prompt (str): Email generation prompt

    Returns:
        str: Generated email text
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in writing marketing emails. Your task is to generate complete, well-structured emails with a clear structure: greeting, main text, conclusion, and signature. Always complete your thoughts and don't cut off text mid-sentence.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=1000,
            presence_penalty=0.6,
            frequency_penalty=0.3,
            stop=None,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error generating email: {str(e)}"


async def get_model_prediction_async(
    client_id: int, model_type: str, token: str
) -> Dict[str, Any]:
    """
    Async version of get_model_prediction.

    Args:
        client_id (int): Client ID (index)
        model_type (str): Type of model to use ("main" or "extended")
        token (str): Authorization token

    Returns:
        Dict[str, Any]: Model prediction result
    """
    try:
        # Get client data
        client_data = get_client_data(client_id)

        if client_data is None:
            return {"error": f"Client with ID {client_id} not found"}

        # Optimize data processing
        row_data = client_data.iloc[0]
        row_json = {
            k: v.isoformat() if isinstance(v, pd.Timestamp) else v
            for k, v in row_data.items()
            if pd.notna(v)
        }

        # Select model URL
        model_url = MAIN_MODEL_URL if model_type == "main" else EXTENDED_MODEL_URL

        # Async request
        async with httpx.AsyncClient(verify=CERT_FILE, timeout=30.0) as client:
            response = await client.post(
                model_url,
                json={"row": row_json},
                headers={"Authorization": f"Bearer {token}"},
            )

            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Model request error: {response.status_code}"}

    except httpx.TimeoutException:
        return {"error": "Request timed out"}
    except Exception as e:
        return {"error": f"Error getting prediction: {str(e)}"}


@st.cache_data
def analyze_all_probabilities(model_type: str = "main") -> List[Dict[str, Any]]:
    """
    Analyze probabilities for all clients using specified model.

    Args:
        model_type (str): Type of model to use ("main" or "extended")

    Returns:
        List[Dict[str, Any]]: List of dictionaries with client IDs and probabilities
    """
    try:
        # Get list of all client IDs and token
        client_ids = get_available_client_ids()
        token = get_auth_token()

        async def process_clients():
            # Create tasks for all clients
            tasks = [
                get_model_prediction_async(client_id, model_type, token)
                for client_id in client_ids
            ]

            # Run all tasks concurrently
            results = await asyncio.gather(*tasks)

            # Process results
            processed_results = []
            for client_id, result in zip(client_ids, results):
                if "Probability" in result:
                    probability = round(result["Probability"], 2)
                    processed_results.append(
                        {"id": client_id, "probability": probability}
                    )

            return processed_results

        # Run async code in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(process_clients())
        loop.close()

        # Sort by probability in descending order
        return sorted(results, key=lambda x: x["probability"], reverse=True)

    except Exception as e:
        return []
