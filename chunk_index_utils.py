"""
Author: Luigi Saetta
Date created: 2024-04-27
Date last modified: 2024-04-27
Python Version: 3.11

Usage: contains the unction to split in chunks and create the index
"""

from glob import glob
from tqdm.auto import tqdm

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from utils import get_console_logger

from config import CHUNK_SIZE, CHUNK_OVERLAP


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


def load_and_rebuild_faiss_index(faiss_dir, books_dir, embed_model):
    """
    load all the books and rebuild the faiss index
    """

    logger = get_console_logger()

    logger.info("local_dir is: %s ...", faiss_dir)
    logger.info("Loading documents from %s...", books_dir)

    text_splitter = get_recursive_text_splitter()

    books_list = glob(books_dir + "/*.pdf")

    docs = []

    for book in tqdm(books_list):
        loader = PyPDFLoader(file_path=book)

        docs += loader.load_and_split(text_splitter=text_splitter)

    logger.info("Loaded %s chunks...", len(docs))

    logger.info("Embedding chunks...")

    v_store = FAISS.from_documents(docs, embed_model)

    logger.info("Saving Vector Store...")
    v_store.save_local(faiss_dir)
