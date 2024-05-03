"""
Author: Luigi Saetta
Date created: 2024-04-27
Date last modified: 2024-04-30
Python Version: 3.11

Usage: contains the functions to split in chunks and create the index
"""

from glob import glob
from tqdm.auto import tqdm

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils import get_console_logger
from config import CHUNK_SIZE, CHUNK_OVERLAP, OPENSEARCH_URL, OPENSEARCH_INDEX_NAME
from config_private import OPENSEARCH_USER, OPENSEARCH_PWD


def get_recursive_text_splitter():
    """
    return a recursive text splitter
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter


def load_book_and_split(book_path):
    """
    load a single book
    """
    logger = get_console_logger()

    text_splitter = get_recursive_text_splitter()

    loader = PyPDFLoader(file_path=book_path)

    docs = loader.load_and_split(text_splitter=text_splitter)

    logger.info("Loaded %s chunks...", len(docs))

    return docs


def add_docs_to_opensearch(docs, embed_model):
    """
    add docs from a book to opensearch vector store
    """
    logger = get_console_logger()

    v_store = OpenSearchVectorSearch(
        embedding_function=embed_model,
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

    logger.info("Saving new documents to Vector Store...")

    v_store.add_documents(docs)

    logger.info("Saved new documents to Vector Store !")


def add_docs_to_faiss(docs, faiss_dir, embed_model):
    """
    add docs from a book to faiss index
    """
    logger = get_console_logger()

    logger.info("Loading Vector Store from local dir %s...", faiss_dir)

    v_store = FAISS.load_local(
        faiss_dir, embed_model, allow_dangerous_deserialization=True
    )

    v_store.add_documents(docs)

    logger.info("Saving Vector Store...")
    v_store.save_local(faiss_dir)


def load_books_and_split(books_dir) -> list:
    """
    load a set of books from books_dir and split in chunks
    """
    logger = get_console_logger()

    logger.info("Loading documents from %s...", books_dir)

    text_splitter = get_recursive_text_splitter()

    books_list = glob(books_dir + "/*.pdf")

    logger.info("Loading books: %s", books_list)

    docs = []

    for book in tqdm(books_list):
        loader = PyPDFLoader(file_path=book)

        docs += loader.load_and_split(text_splitter=text_splitter)

    logger.info("Loaded %s chunks...", len(docs))

    return docs


def load_and_rebuild_faiss_index(faiss_dir, books_dir, embed_model):
    """
    load all the books and rebuild the faiss index
    """

    logger = get_console_logger()

    logger.info("local_dir is: %s ...", faiss_dir)

    docs = load_books_and_split(books_dir)

    logger.info("Embedding chunks...")

    v_store = FAISS.from_documents(docs, embed_model)

    logger.info("Saving Vector Store...")
    v_store.save_local(faiss_dir)

    return v_store
