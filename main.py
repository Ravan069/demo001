import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

# FastAPI app initialization


# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GOOGLE_API_KEY")
if gemini_api_key is None:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

genai.configure(api_key=gemini_api_key)

# Function to extract text from a static PDF
def get_pdf_text(pdf_file_path):
    text = ""
    pdf_reader = PdfReader(pdf_file_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(raw_text)

# Function to convert chunks into vector embeddings
def get_vector(chunks):
    if not chunks:
        return None
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=gemini_api_key)
    return FAISS.from_texts(texts=chunks, embedding=embeddings)

# Function to handle conversation
def conversation_chain():
    template = """
    Answer the asked question in detail.
    Context: \n{context}\n
    Question: \n{question}\n
    Answer:
    """
    model_instance = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=gemini_api_key)
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return load_qa_chain(model_instance, chain_type="stuff", prompt=prompt), model_instance

# Function to process the user's question
def user_question(question, db, chain, raw_text):
    if db is None:
        return "Please process the PDF first."
    docs = db.similarity_search(question, k=5)
    response = chain.invoke({"input_documents": docs, "question": question, "context": raw_text}, return_only_outputs=True)
    return response.get("output_text")

    
def main():
    # Set the page configuration
    st.set_page_config(page_title="RAG Application", page_icon="🤖", layout="wide")
    st.header("RAG Application")

    # Initialize session state variables specific to this page
    if "messages_chatbot_2" not in st.session_state:
        st.session_state.messages_chatbot_2 = []
    if "vector_store_chatbot_2" not in st.session_state:
        st.session_state.vector_store_chatbot_2 = None
    if "chain_chatbot_2" not in st.session_state:
        st.session_state.chain_chatbot_2 = None
    if "raw_text_chatbot_2" not in st.session_state:
        st.session_state.raw_text_chatbot_2 = None
        
    pdf_docs = "c.pdf"


    raw_text = get_pdf_text(pdf_docs)
                        
    chunks = get_text_chunks(raw_text)
    vector_store = get_vector(chunks)
    chain, _ = conversation_chain()

    if vector_store and chain and raw_text:
        st.session_state.vector_store_chatbot_2 = vector_store
        st.session_state.chain_chatbot_2 = chain
        st.session_state.raw_text_chatbot_2 = raw_text 
    
    for message in st.session_state.messages_chatbot_2:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input for user questions
    if st.session_state.vector_store_chatbot_2 and st.session_state.chain_chatbot_2 and st.session_state.raw_text_chatbot_2:
        user_query = st.chat_input("Ask your question:")
        if user_query:
            with st.chat_message("user"):
                st.markdown(user_query)
            st.session_state.messages_chatbot_2.append({"role": "user", "content": user_query})

            response = user_question(user_query, st.session_state.vector_store_chatbot_2, st.session_state.chain_chatbot_2, st.session_state.raw_text_chatbot_2)
            if response:
                st.session_state.messages_chatbot_2.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)                            
                                
                                
if __name__ == '__main__':
    main()
                                          