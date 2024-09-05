import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
from dotenv import load_dotenv
import tempfile

load_dotenv()
groq_api_key = os.environ['GROQ_API_KEY']

st.title("DocBot: AI-Powered PDF Assistant")

# Upload PDF file
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Create a temporary file
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file_path = os.path.join(tmp_dir, "temp.pdf")
        with open(tmp_file_path, "wb") as tmp_file:
            # Write the contents of the uploaded file to the temporary file
            tmp_file.write(uploaded_file.read())

            # Load PDF and process
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            final_documents = text_splitter.split_documents(docs)
            
            # Create vectors from the documents
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectors = FAISS.from_documents(final_documents, embeddings)
            retriever = vectors.as_retriever()
            
            # Define LLM and prompt template
            llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")
            prompt = ChatPromptTemplate.from_template(
                """
                Answer the questions based on the provided context only.
                Please provide the most accurate response based on the question
                <context>
                {context}
                <context>
                Questions:{input}
                """
            )
            document_chain = create_stuff_documents_chain(llm, prompt)
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            # Input prompt from user
            user_prompt = st.text_input("Enter your question here")
            if user_prompt:
                # start = time.process_time()
                response = retrieval_chain.invoke({"input": user_prompt})
                # response_time = time.process_time() - start
                # st.write(f"Response time: {response_time:.2f} seconds")
                st.write(response['answer'])

                # With a Streamlit expander
                # with st.expander("Document Similarity Search"):
                #     # Find the relevant chunks
                #     for i, doc in enumerate(response.get("context", [])):
                #         st.write(doc.page_content)