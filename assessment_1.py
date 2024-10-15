__import__('pysqlite3')
import sys
import sqlite3
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
from utilities.ai_embedding import text_small_embedding  
from utilities.ai_inference import gpt4o_mini_inference  
from utilities.chroma_db import (
    get_or_create_persistent_chromadb_client_and_collection,
    add_document_chunk_to_chroma_collection,
    query_chromadb_collection,
    delete_chromadb_collection
)
from utilities.documents import upload_document, read_document, chunk_document, delete_document
from utilities.layout import page_config
import os


page_config()

client, collection = get_or_create_persistent_chromadb_client_and_collection("legal_docs_collection")

# Step 1: Upload and process the document
st.subheader("Step 1: Upload Legal Document")


document_folder = "uploaded_files"
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

st.session_state['uploaded_file'] = st.session_state.get('uploaded_file', None)
st.session_state['query_result'] = st.session_state.get('query_result', None)
st.session_state['system_prompt'] = st.session_state.get(
    'system_prompt', "You are an expert lawyer, who specialises in Australian contract law."
)
st.session_state['file_deleted'] = st.session_state.get('file_deleted', False)

# Define a cached function to chunk and cache document data
@st.cache_data
def get_document_chunks(document_folder, document_name):
    return chunk_document(document_folder, document_name)

if uploaded_file:
    if not os.path.exists(document_folder):
        os.makedirs(document_folder)

    document_name = uploaded_file.name
    file_path = os.path.join(document_folder, document_name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Document {document_name} uploaded successfully!")

    # Store chunks and embedding vectors only on the first upload and chunking
    if "chunks" not in st.session_state:
        document_text = read_document(document_folder, document_name)
        if document_text:
            st.write("Document read successfully!")

            # Generate document summary using AI
            summary_prompt = f"Please summarize the following document in under 300 characters:\n\n{document_text}"
            document_summary = gpt4o_mini_inference(summary_prompt, "Summarize the document")
            
            if document_summary:
                st.write("Document Summary:")
                st.write(document_summary)
            else:
                st.write("Document Preview:")
                st.write(document_text[:300])

            st.session_state.chunks = get_document_chunks(document_folder, document_name)
            st.session_state.current_chunk_index = 0

            for chunk in st.session_state.chunks:
                add_document_chunk_to_chroma_collection("legal_docs_collection", chunk)

# Pagination feature for document chunks
if "chunks" in st.session_state:
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Previous") and st.session_state.current_chunk_index > 0:
            st.session_state.current_chunk_index -= 1
    with col2:
        if st.button("Next") and st.session_state.current_chunk_index < len(st.session_state.chunks) - 1:
            st.session_state.current_chunk_index += 1

    current_chunk = st.session_state.chunks[st.session_state.current_chunk_index]
    st.write(f"Chunk {st.session_state.current_chunk_index + 1} of {len(st.session_state.chunks)}")
    st.write(current_chunk)

    # Delete button
    delete_document(document_folder, document_name)


# Step 2: User query
st.subheader("Step 2: Ask a Question")
user_query = st.text_input("Enter your legal question:", key="user_query_input")

if st.button("Submit Query"):
    if user_query:
        # Return an adaptive number of chunks based on the query length
        query_length = len(user_query.split())
        if query_length < 5:
            n_results = 1  
        elif query_length < 20:
            n_results = 3  
        else:
            n_results = 5  

        st.session_state.query_result = query_chromadb_collection(
            "legal_docs_collection", user_query, n_results=n_results
        )

        if st.session_state.query_result:
            st.write(f"Displaying top {n_results} relevant document chunks:")
            for i, result in enumerate(st.session_state.query_result):
                st.write(f"Chunk {i + 1}: {result}")
        else:
            st.write("No relevant information found.")
    else:
        st.write("Please enter a query.")


# Step 3: Generate answer based on retrieval results
st.subheader("Step 3: Generating Answer")

if st.session_state.query_result:
    st.write("Sending the following document chunks to GPT:")
    document_chunks_str = "\n".join(st.session_state.query_result)

    generated_answer = gpt4o_mini_inference(user_query, document_chunks_str)
    if generated_answer:
        st.write(f"Generated Answer: {generated_answer}")
    else:
        st.error("No answer generated. Check the GPT API connection and input.")
