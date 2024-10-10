import streamlit as st
from utilities.ai_embedding import text_small_embedding
from utilities.ai_inference import gpt4o_mini_inference, gpt4o_mini_inference_yes_no
from utilities.chroma_db import get_or_create_persistent_chromadb_client_and_collection, add_document_chunk_to_chroma_collection, query_chromadb_collection, delete_chromadb_collection
from utilities.documents import upload_document, read_document, chunk_document, download_document, delete_document
from utilities.layout import page_config
import os

client, collection = get_or_create_persistent_chromadb_client_and_collection("legal_docs_collection") 

# Step 1: Upload and process document
st.subheader("Step 1: Upload Legal Document")

# 定义文件存储文件夹
document_folder = "uploaded_files"

# 上传文件
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

if "query_result" not in st.session_state:
    st.session_state.query_result = None

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "You are an expert lawyer, who specialises in Australian contract law."

if "file_deleted" not in st.session_state:
    st.session_state.file_deleted = False

if uploaded_file:
    # 检查并创建存储文件夹
    if not os.path.exists(document_folder):
        os.makedirs(document_folder)

    # 获取文件名并定义完整路径
    document_name = uploaded_file.name
    file_path = os.path.join(document_folder, document_name)
    
    # 将文件保存到本地
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"Document {document_name} uploaded successfully!")
    
    # 调用 read_document() 读取文件
    document_text = read_document(document_folder, document_name)
    
    if document_text:
        st.write("Document read successfully!")
        st.write(document_text[:500])


# 对文档进行分块
        if "chunks" not in st.session_state:
            st.session_state.chunks = chunk_document(document_folder, document_name)
            st.session_state.current_chunk_index = 0  # 初始化索引
        
        chunks = st.session_state.chunks
        if chunks:
            st.write(f"Document split into {len(chunks)} chunks.")
            # 将文档块添加到 ChromaDB 集合中
            for chunk in chunks:
                add_document_chunk_to_chroma_collection("legal_docs_collection", chunk)

            col1, col2 = st.columns([1, 1]) 
            
            with col1:
                if st.button("Previous") and st.session_state.current_chunk_index > 0:
                    st.session_state.current_chunk_index -= 1

            with col2:
                if st.button("Next") and st.session_state.current_chunk_index < len(chunks) - 1:
                    st.session_state.current_chunk_index += 1
            
            current_chunk = chunks[st.session_state.current_chunk_index]
            st.write(f"Chunk {st.session_state.current_chunk_index + 1} of {len(chunks)}")
            st.write(current_chunk)
        else:
            st.error("Document could not be chunked.")

# 显示文件内容
    document_text = read_document(document_folder, document_name)
    st.write(document_text[:500])  # 显示前500个字符
    
    # 删除按钮
    delete_document(document_folder, document_name)
    
# Step 2: 用户查询
st.subheader("Step 2: Ask a Question")
user_query = st.text_input("Enter your legal question:", key="user_query_input")  # 添加唯一的key

if st.button("Submit Query"):
    if user_query:
        n_results = 5  # 可以根据需要调整返回的结果数量
        st.session_state.query_result = query_chromadb_collection("legal_docs_collection", user_query, n_results=n_results)  # 添加 n_results 参数
        
        if st.session_state.query_result:
            st.write("Relevant document chunks found:")
            for i, result in enumerate(st.session_state.query_result):
                st.write(f"Chunk {i+1}: {result}")
        else:
            st.write("No relevant information found.")
    else:
        st.write("Please enter a query.")

# Step 3: 基于检索结果生成答案
st.subheader("Step 3: Generating Answer")

if st.session_state.query_result:
   
    st.write("Sending the following document chunks to GPT:")
    
    st.write(f"User query: {user_query}")  # 输出用户的问题

    document_chunks_str = "\n".join(st.session_state.query_result) # 用换行符拼接文档块 # 调用 GPT 接口生成答案 
    # 调用 GPT 接口生成答案
    generated_answer = gpt4o_mini_inference(user_query, document_chunks_str)
    
    if generated_answer:
        st.write(f"Generated Answer: {generated_answer}")
    else:
        st.error("No answer generated. Check the GPT API connection and input.")


