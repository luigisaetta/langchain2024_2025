"""
Author: Luigi Saetta
Date created: 2024-04-27
Date last modified: 2024-04-30
Python Version: 3.11
"""

from glob import glob
from tqdm.auto import tqdm

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import OpenSearchVectorSearch

from factory import get_embed_model
from chunk_index_utils import get_recursive_text_splitter
from utils import get_console_logger

from config import BOOKS_DIR, OPENSEARCH_URL
from config_private import OPENSEARCH_PWD

logger = get_console_logger()

logger.info("Loading documents from %s...", BOOKS_DIR)

text_splitter = get_recursive_text_splitter()

books_list = glob(BOOKS_DIR + "/*.pdf")

docs = []

for book in tqdm(books_list):
    loader = PyPDFLoader(file_path=book)

    docs += loader.load_and_split(text_splitter=text_splitter)

logger.info("Loaded %s chunks...", len(docs))

embed_model = get_embed_model(model_type="OCI")

docsearch = OpenSearchVectorSearch.from_documents(
    docs,
    embedding=embed_model,
    opensearch_url=OPENSEARCH_URL,
    http_auth=("lsaetta", OPENSEARCH_PWD),
    use_ssl=True,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
    bulk_size=5000,
    index_name="test1",
    engine="faiss",
)

# test
QUERY = "La metformina può essere usata per curare il diabete di tipo 2 nei pazienti anziani?"
results = docsearch.similarity_search(QUERY, k=4)

print(len(results))
print(results)
