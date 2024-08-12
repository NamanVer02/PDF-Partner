import streamlit as st
from PyPDF2 import PdfReader
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

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
    embeddings = HuggingFaceEmbeddings()
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
    return retrieval_chain.invoke(query)


st.set_page_config(
    page_title="PDF Partner"
)
st.markdown("<h1 style='text-align: center; font-size:5rem;'>PDF Partner</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:1rem; font-weight: 200;'>Turn your boring PDF's into an interactive Q and A</p><br><br>", unsafe_allow_html=True)


uploaded_file = st.file_uploader(
    "Upload a .pdf document to ask questions about it",
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
            "Going through your document"
            ):
            response = generate_response(
                uploaded_file,
                query_text
            )
            result.append(response)
            
if len(result):
    st.info(response['result'])
    
st.markdown("<a style='text-align: center; font-size:0.7rem; font-weight: 200;' href='https://www.namanverma.in/'>Made by Naman Vemra</a><br><br>", unsafe_allow_html=True)