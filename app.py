import streamlit as st
import os
from backend import process_and_index_documents, answer_question

st.set_page_config(
    page_title="Document QA System",
    page_icon="📄",
    layout="wide"
)


UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


st.title("📄 Document Question Answering System")

st.sidebar.header("📂 Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload your PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    st.sidebar.write("### Uploaded Files:")
    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        st.sidebar.write(f"✔️ {uploaded_file.name}")
    
    process_and_index_documents(uploaded_files)


st.markdown("""
### How to Use:
1. Upload your documents using the sidebar.
2. Select a document to view its content.
3. Ask a question in the input box, and the system will provide an AI-generated answer based on the selected document.
""")

question = st.text_input("Ask a question:")

if question:
    answer = answer_question(question)
    st.write("### Answer:")
    st.write(answer)
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit, Pinecone, Langchain, and Python.")

# in backend.py file 

# python -m streamlit run app.py
# Sure! To ensure that the Pinecone API key is correctly inserted before starting the program, 
# you should initialize the Pinecone API at the beginning of your code. 
# Here's how you can include your Pinecone API key:

# Insert Pinecone API Key:
# In the code above, 
# replace the placeholder "your-pinecone-api-key" with your actual Pinecone API key. 
# You can get this key from your Pinecone account