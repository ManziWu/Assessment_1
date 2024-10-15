import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import uuid
import os
from dotenv import load_dotenv
from utilities.ai_embedding import text_small_embedding  # 导入嵌入向量生成函数

# 加载环境变量，获取 API 密钥
load_dotenv()


# 步骤 1：初始化数据库客户端和集合
def get_or_create_persistent_chromadb_client_and_collection(collection_name):
    # 创建持久化客户端，将数据存储到指定路径中
    chroma_client = chromadb.PersistentClient(f"./data/chromadb/{collection_name}")

    # 创建或获取集合，指定使用 OpenAI 嵌入向量模型和余弦相似度空间
    collection = chroma_client.create_collection(
        name=collection_name,
        embedding_function=embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        ),
        metadata={"hnsw:space": "cosine"},
        get_or_create=True
    )
    return collection, chroma_client


# 步骤 2：在分块时生成嵌入向量并存储
def add_document_chunk_to_chroma_collection(collection_name, document_chunk, document_id=None):
    collection, chroma_client = get_or_create_persistent_chromadb_client_and_collection(collection_name)

    # 若未提供 document_id，则生成一个唯一 ID
    document_id = document_id or str(uuid.uuid4())

    # 生成文档块的嵌入向量
    chunk_embedding = text_small_embedding(document_chunk)  # 生成嵌入向量

    # 将文档块、ID 和嵌入向量存储到数据库中
    collection.add(
        documents=[document_chunk],
        ids=[document_id],
        embeddings=[chunk_embedding]  # 存储嵌入向量
    )


# 步骤 3：在查询时生成查询嵌入向量并进行语义搜索
def query_chromadb_collection(collection_name, query, n_results):
    collection, chroma_client = get_or_create_persistent_chromadb_client_and_collection(collection_name)

    # 生成查询的嵌入向量
    query_embedding = text_small_embedding(query)  # 将查询转为嵌入向量

    # 使用嵌入向量进行语义搜索
    documents = collection.query(
        query_embeddings=[query_embedding],  # 使用查询的嵌入向量
        include=["documents", "metadatas"],
        n_results=n_results
    )

    # 检查查询结果并返回最相关的文档块
    if documents and documents['documents'] and documents['documents'][0]:
        return documents['documents'][0]
    else:
        return []


# 删除指定集合
def delete_chromadb_collection(collection_name):
    collection, chroma_client = get_or_create_persistent_chromadb_client_and_collection(collection_name)

    try:
        # 尝试删除集合
        chroma_client.delete_collection(name=collection_name)
        return f"Collection '{collection_name}' has been successfully deleted."
    except Exception as e:
        print(f"Error: {e}")
        return f"An error occurred while trying to delete the collection: {e}"
