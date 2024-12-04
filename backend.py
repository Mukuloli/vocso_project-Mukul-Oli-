
import os
import openai
from pinecone import Pinecone, ServerlessSpec
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as LangChainPinecone
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI as LangChainOpenAI
from langchain.document_loaders import PyPDFLoader  
from langchain.text_splitter import CharacterTextSplitter
import PyPDF2
from langchain.schema import Document  

os.environ["PINECONE_API_KEY"] = "YOUR_PINECONE_API_KEY"
openai.api_key = "YOUR_OPENAI_API_KEY"


pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"], environment="us-west1-gcp")  

index_name = "INDEX_NAME"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-west1')
    )

def process_pdf_file(file_path):
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text


def process_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def process_document(file_path):
    if file_path.endswith(".txt"):
        return process_text_file(file_path)
    elif file_path.endswith(".pdf"):
        return process_pdf_file(file_path)
    else:
        raise ValueError("Unsupported file type")

def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=100):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)

def index_document_in_pinecone(text, document_name):
    split_docs = split_text_into_chunks(text)
    documents = [Document(page_content=chunk) for chunk in split_docs]
    
    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
    pinecone_vector_store = LangChainPinecone.from_documents(
        documents, embeddings, index_name=index_name
    )
    return pinecone_vector_store
embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
retriever = LangChainPinecone.from_existing_index(index_name=index_name, embedding=embeddings)
openai.api_key = "OPEAN_API_KEY"
embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
index_name = "samsung"
retriever = LangChainPinecone.from_existing_index(index_name=index_name, embedding=embeddings)
llm = LangChainOpenAI(openai_api_key=openai.api_key)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,   
    retriever=retriever.as_retriever()   
)

chat_history = []

def query_documents_from_pinecone(query):
    response = qa_chain.run({"question": query, "chat_history": chat_history})
    chat_history.append((query, response))
    return response

def process_and_index_documents(uploaded_files):
    for uploaded_file in uploaded_files:
        file_path = os.path.join("uploads", uploaded_file.name)
        document_text = process_document(file_path)
        index_document_in_pinecone(document_text, uploaded_file.name)

def answer_question(query):
    return query_documents_from_pinecone(query)

