import streamlit as st
from PyPDF2 import PdfReader
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

def generate_response(file, query):
    #format file
    reader = PdfReader(file)
    formatted_document = []
    for page in reader.pages:
        formatted_document.append(page.extract_text())
    #split file
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )
    docs = text_splitter.create_documents(formatted_document)
    #create embeddings
    embeddings = SentenceTransformerEmbeddings()
    #load to vector database
    #store = Chroma.from_documents(texts, embeddings)

    store = FAISS.from_documents(docs, embeddings)
    
    #create retrieval chain
    llm = ChatGroq(
        temperature=0.2,
        model="llama3-70b-8192",
        api_key=st.secrets["GROQ_API_KEY"]
    )
    
    retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=store.as_retriever()
    )
    #run chain with query
    return retrieval_chain.run(query)


st.set_page_config(
    page_title="Q&A from a long PDF Document"
)
st.title("Q&A from a long PDF Document")

uploaded_file = st.file_uploader(
    "Upload a .pdf document",
    type="pdf"
)

query_text = st.text_input(
    "Enter your question:",
    placeholder="Write your question here",
    disabled=not uploaded_file
)

result = []
with st.form(
    "myform",
    clear_on_submit=True
):
    submitted = st.form_submit_button(
        "Submit",
        disabled=not (uploaded_file and query_text)
    )
    if submitted:
        with st.spinner(
            "Wait, please. I am working on it..."
            ):
            response = generate_response(
                uploaded_file,
                query_text
            )
            result.append(response)
            
if len(result):
    st.info(response)