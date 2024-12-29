import streamlit as st
import os
from dotenv import load_dotenv
from file_handler import FileHandler
from chat_handler import ChatHandler

# Load environment variables
load_dotenv()

# Initialize Handlers
VECTOR_DB_PATH = st.secrets["VECTOR_DB_PATH_DB"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
os.makedirs(VECTOR_DB_PATH, exist_ok=True)

file_handler = FileHandler(VECTOR_DB_PATH,OPENAI_API_KEY)
chat_handler = ChatHandler(VECTOR_DB_PATH,OPENAI_API_KEY)

# Streamlit UI
st.set_page_config(layout="wide", page_title="AI Connect - Smarter Network Planning for the Future")
st.title("Chatbot - Smarter Network Planning for the Future")
# Enable the below line to show the sidebar

# Left Side: File Upload
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

# Main: Chat Interface
# st.header("Chat Interface")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat messages from history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Type your question here..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    # Generate AI response
    with st.spinner("Processing your question..."):
        response = chat_handler.answer_question(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state["messages"].append({"role": "assistant", "content": response})
