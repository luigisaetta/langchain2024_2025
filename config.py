"""
Author: Luigi Saetta
Date created: 2024-04-27
Date last modified: 2024-04-27
Python Version: 3.11
"""

# per ora usiamo il tokenizer di Cohere...
TOKENIZER = "Cohere/Cohere-embed-multilingual-v3.0"

# title for the UI
TITLE = "AI Assistant with LangChain 🦜"
HELLO_MSG = "Ciao, come posso aiutarti?"

ADD_REFERENCES = True
VERBOSE = False

# enable tracing with LangSmith
ENABLE_TRACING = True

# for chunking
# in chars
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 50

# OCI GenAI model used for Embeddings
# to batch embedding with OCI
# with Cohere embeddings max is 96
# value: COHERE, OCI
EMBED_MODEL_TYPE = "OCI"
EMBED_BATCH_SIZE = 90
OCI_EMBED_MODEL = "cohere.embed-multilingual-v3.0"
COHERE_EMBED_MODEL = "embed-multilingual-v3.0"

ENDPOINT = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"

# reranker
ADD_RERANKER = True
COHERE_RERANKER_MODEL = "rerank-multilingual-v3.0"

# retriever
TOP_K = 6
TOP_N = 4

# Oracle VS
EMBEDDINGS_BITS = 32

# Vector Store
# VECTOR_STORE_TYPE = "FAISS"
VECTOR_STORE_TYPE = "OPENSEARCH"
# VECTOR_STORE_TYPE = "23AI"

# OPENSEARCH
# using local as docker
OPENSEARCH_URL = "https://localhost:9200"
OPENSEARCH_INDEX_NAME = "test1"

# 23AI
COLLECTION_NAME = "ORACLE_KNOWLEDGE"

# parametri per leggere il database FAISS
BOOKS_DIR = "./books"
# la directory in cui il vector store è salvato
FAISS_DIR = "./faiss_index"

# COHERE, OCI
LLM_MODEL_TYPE = "COHERE"
# Cohere params
COHERE_GENAI_MODEL = "command-r"

# params for LLM
TEMPERATURE = 0.1
MAX_TOKENS = 1024

# to enable streaming
DO_STREAMING = True

# for TRACING
LANGCHAIN_PROJECT = "memory05052024-1"

# Opensearch shared params
OPENSEARCH_SHARED_PARAMS = {
    "opensearch_url": OPENSEARCH_URL,
    "use_ssl": True,
    "verify_certs": False,
    "ssl_assert_hostname": False,
    "ssl_show_warn": False,
    "bulk_size": 5000,
    "index_name": OPENSEARCH_INDEX_NAME,
    "engine": "faiss",
}
