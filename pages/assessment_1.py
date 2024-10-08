import streamlit as st
from utilities.ai_embedding import text_small_embedding
from utilities.ai_inference import gpt4o_mini_inference, gpt4o_mini_inference_yes_no
from utilities.chroma_db import get_or_create_persistent_chromadb_client_and_collection, add_document_chunk_to_chroma_collection, query_chromadb_collection, delete_chromadb_collection
from utilities.documents import upload_document, read_document, chunk_document, download_document, delete_document
from utilities.layout import page_config

client, collection = get_or_create_persistent_chromadb_client_and_collection("legal_docs_collection")

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

if "query_result" not in st.session_state:
    st.session_state.query_result = None

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "You are an expert lawyer, who specialises in Australian contract law."

st.subheader("Step 1: Upload Legal Document")
uploaded_file = st.file_uploader("Choose a file", type=["pdf"])

if uploaded_file:
    st.session_state.uploaded_file = uploaded_file
    document_name = uploaded_file.name  
    document_text = read_document(uploaded_file, document_name)  
    st.write("Document uploaded successfully!")
    
    chunks = chunk_document(document_text)
    st.write(f"Document split into {len(chunks)} chunks.")
    
    for chunk in chunks:
        add_document_chunk_to_chroma_collection(collection, chunk)
    st.write("Document chunks added to the vector database.")


st.subheader("Step 2: Ask a Question")
user_query = st.text_input("Enter your legal question:")

if st.button("Submit Query"):
    if user_query and st.session_state.uploaded_file:
        
        query_result = query_chromadb_collection(collection, user_query)
        
        if query_result:
            st.session_state.query_result = query_result
            st.write("Relevant document chunks found:")
            for i, result in enumerate(query_result):
                st.write(f"Result {i+1}: {result}")
                
            st.subheader("Step 3: Generating Answer")
            generated_answer = gpt4o_mini_inference(user_query, query_result)
            st.write(f"Answer: {generated_answer}")
        else:
            st.write("No relevant information found.")
    else:
        st.write("Please upload a document and enter a query.")
