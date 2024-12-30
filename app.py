import streamlit as st
import os
from dotenv import load_dotenv
from file_handler import FileHandler
from chat_handler import ChatHandler

# Load environment variables
load_dotenv()

# Static credentials
USERNAME = st.secrets["USERNAME"]
PASSWORD = st.secrets["PASSWORD"]

# Initialize Handlers
VECTOR_DB_PATH = st.secrets["VECTOR_DB_PATH_DB"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
HUGGINGFACE_API_TOKEN = st.secrets["HUGGINGFACE_API_TOKEN"]

os.makedirs(VECTOR_DB_PATH, exist_ok=True)

file_handler = FileHandler(VECTOR_DB_PATH, HUGGINGFACE_API_TOKEN)
chat_handler = ChatHandler(VECTOR_DB_PATH, HUGGINGFACE_API_TOKEN, OPENAI_API_KEY)

# Streamlit UI
st.set_page_config(layout="wide", page_title="AI Connect - Smarter Network Planning for the Future")

# Session state to track login status
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# Login page
# Refined Login Page
if not st.session_state["logged_in"]:
    # Customize page title
    st.markdown(
        """
        <style>
        .title {
            font-size: 2.5rem;
            color: #1f77b4;
            font-weight: bold;
            text-align: center;
            margin-bottom: 10px;
        }
        .subtitle {
            font-size: 1.2rem;
            color: #555;
            text-align: center;
            margin-bottom: 20px;
        }
        .login-box {
            margin: auto;
            width: 50%;
            padding: 20px;
            background: #f9f9f9;
            border: 1px solid #ddd;
            border-radius: 10px;
        }
        .login-box input {
            margin-bottom: 10px;
        }
        </style>
        <div>
            <div class="title">Welcome to AI Connect</div>
            <div class="subtitle">Smarter Network Planning for the Future</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Centered Login Box
    # st.markdown('<div class="login-box">', unsafe_allow_html=True)
    st.subheader("Login to Continue")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.session_state["logged_in"] = True
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password.")
    st.markdown("</div>", unsafe_allow_html=True)
else:
    # Main app (Chat Interface)
    st.title("Chatbot - Smarter Network Planning for the Future")
    st.sidebar.header("Upload Documents")
    uploaded_file = st.sidebar.file_uploader("Upload PDF, Excel, Docx, or Txt", type=["pdf", "xlsx", "docx", "txt", "csv"])
    document_name = st.sidebar.text_input("Document Name", "")
    document_description = st.sidebar.text_area("Document Description", "")

    if st.sidebar.button("Process File"):
        if uploaded_file:
            with st.spinner("Processing your file..."):
                response = file_handler.handle_file_upload(
                    file=uploaded_file,
                    document_name=document_name,
                    document_description=document_description,
                )
                st.sidebar.success(f"File processed: {response['message']}")
        else:
            st.sidebar.warning("Please upload a file before processing.")

    # Chat Interface
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display chat messages from history
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Type your question here..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state["messages"].append({"role": "user", "content": prompt})

        with st.spinner("Processing your question..."):
            response = chat_handler.answer_question(prompt)
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state["messages"].append({"role": "assistant", "content": response})

# Logout button
if st.session_state["logged_in"]:
    if st.sidebar.button("Logout"):
        st.session_state["logged_in"] = False
        st.rerun()
