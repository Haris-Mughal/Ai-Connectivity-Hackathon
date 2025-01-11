# AI Connect - Smarter Network Planning for the Future

This project is a Retrieval-Augmented Generation (RAG) system designed to assist users in querying datasets related to 5G network optimization. The system leverages a vector database for dataset embeddings and a conversational AI assistant powered by Groq's Llama-3.1 open-source instructor model.
## Features

- Preloaded vector database with embedded Network datasets.
- Query energy datasets using a conversational AI assistant.
- Conversational responses powered by Groq's Llama-3.1 open-source model.
- User-friendly interface built with Streamlit.

## Installation

### Prerequisites

1. **Python 3.10** or higher
2. **Pip** or a compatible package manager
3. A **Groq API** Key to access the **Llama-3.1 model**

### Setup Instructions

1. **Clone the repository:**

   ```bash
    git clone https://github.com/rajeshthangaraj1/ai-connectivity-hackathon.git
    cd ai-connectivity-hackathon
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

4. **Set up the `.env` file:**
   In the root directory, locate the .env.example file, rename it to .env, and update it with the necessary credentials or environment variables:

    ```env
    GROQ_API_KEY = "your-groq-api-key-here"
    LOG_PATH = "./logs"
    HUGGINGFACE_API_TOKEN="your-huggingface-api-token"
    USERNAME="your-username-here"
    PASSWORD="your-password-here"
    MILVUS_HOST="your-milvus-host"  
    MILVUS_PORT="your-milvus-port"
    ```
   

### Usage

1. **Start the Streamlit application:**

   ```bash
    streamlit run app.py
    ```

2. **Ask questions in the chat interface on the right side of the screen.**

   - Example of a question: **What are the potential inefficiencies in the current 5G network deployment, and how can we optimize resource usage to improve latency and throughput?.**





