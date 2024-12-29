# AI Connect - Smarter Network Planning for the Future

This project is a Retrieval-Augmented Generation (RAG) system that allows users to query embedded datasets related to 5G network optimization. The system is preloaded with 5G network datasets in a vector database and provides a conversational AI assistant powered by the OpenAI Model

## Features

- Preloaded vector database with embedded Network datasets.
- Query energy datasets using a conversational AI assistant.
- Supports embedding generation using OpenAI's GPT-4 model.
- User-friendly interface built with Streamlit.

## Installation

### Prerequisites

1. **Python 3.10** or higher
2. **Pip** or a compatible package manager
3. An **OpenAI API Key**

### Setup Instructions

1. **Clone the repository:**

   ```bash
    git clone https://github.com/rajeshthangaraj1/ai-connectivity-hackathon.git
    cd doge-hackathon-rag
    ```
   
2. **Create a virtual environment:**

   ```bash
    python -m venv venv
    ```

    - **For Linux/Mac:**

        ```bash
        source venv/bin/activate
        ```

    - **For Windows:**

        ```bash
        venv\Scripts\activate
        ```
  
3. **Install dependencies:**
   ```bash
    pip install -r requirements.txt
    ```

4. **Set up the `.streamlit/secrets.toml` file:**
   Create a secrets.toml file inside the .streamlit folder in the root directory with the following variables:

    ```env
    OPENAI_API_KEY = "your-openai-api-key-here"
    VECTOR_DB_PATH_DB = "./vectordb"

### Usage

1. **Start the Streamlit application:**

   ```bash
    streamlit run app.py
    ```

2. **Ask questions in the chat interface on the right side of the screen.**

   - Example of a question: **What are the potential inefficiencies in the current 5G network deployment, and how can we optimize resource usage to improve latency and throughput?.**





