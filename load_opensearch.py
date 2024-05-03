"""
Author: Luigi Saetta
Date created: 2024-04-27
Date last modified: 2024-04-30
Python Version: 3.11

Usage:
    With this code you can load the initial content inside 
    the OpenSearch based Vector Store
"""

from langchain_community.vectorstores import OpenSearchVectorSearch

from factory import get_embed_model
from chunk_index_utils import load_books_and_split
from utils import get_console_logger

from config import BOOKS_DIR, OPENSEARCH_URL, OPENSEARCH_INDEX_NAME
from config_private import OPENSEARCH_USER, OPENSEARCH_PWD

logger = get_console_logger()

# load all the books in BOOKS_DIR
docs = load_books_and_split(BOOKS_DIR)

embed_model = get_embed_model(model_type="OCI")

# load text and embeddings in OpenSearch
docsearch = OpenSearchVectorSearch.from_documents(
    docs,
    embedding=embed_model,
    opensearch_url=OPENSEARCH_URL,
    http_auth=(OPENSEARCH_USER, OPENSEARCH_PWD),
    use_ssl=True,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
    bulk_size=5000,
    index_name=OPENSEARCH_INDEX_NAME,
    engine="faiss",
)

# Do a test
QUERY = "La metformina può essere usata per curare il diabete di tipo 2 nei pazienti anziani?"
results = docsearch.similarity_search(QUERY, k=4)

print("Test:")
print(QUERY)
print(len(results))
print(results)
