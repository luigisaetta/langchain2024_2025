"""
Author: Luigi Saetta
Date created: 2024-05-20
Date last modified: 2024-05-20

Usage:
    This module handles the creation of the Vector Store 
    used in the RAG chain, vased on config

Python Version: 3.11
"""

import os
import logging
import oracledb

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_community.vectorstores import Qdrant
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_community.vectorstores.utils import DistanceStrategy

# Qdrant
from qdrant_client import QdrantClient

from utils import check_value_in_list
from chunk_index_utils import load_and_rebuild_faiss_index

from config import (
    COLLECTION_NAME,
    QDRANT_URL,
    # shared params for opensearch
    OPENSEARCH_SHARED_PARAMS,
)
from config_private import (
    OPENSEARCH_USER,
    OPENSEARCH_PWD,
    DB_USER,
    DB_PWD,
    DB_HOST_IP,
    DB_SERVICE,
)


def get_vector_store(vector_store_type, embed_model, local_index_dir, books_dir):
    """
    Read or rebuild the index and retur a Vector Store
    """
    logger = logging.getLogger("ConsoleLogger")

    check_value_in_list(vector_store_type, ["FAISS", "OPENSEARCH", "23AI", "QDRANT"])

    if vector_store_type == "FAISS":
        if os.path.exists(local_index_dir):
            logger.info("Loading Vector Store from local dir %s...", local_index_dir)

            v_store = FAISS.load_local(
                local_index_dir, embed_model, allow_dangerous_deserialization=True
            )
            logger.info("Loaded %s chunks of text !!!", v_store.index.ntotal)
        else:
            v_store = load_and_rebuild_faiss_index(
                local_index_dir, books_dir, embed_model
            )

    if vector_store_type == "OPENSEARCH":
        # this assumes that there is an OpenSearch cluster available
        # or docker
        # at the specified URL
        # data already loaded in
        v_store = OpenSearchVectorSearch(
            embedding_function=embed_model,
            http_auth=(OPENSEARCH_USER, OPENSEARCH_PWD),
            **OPENSEARCH_SHARED_PARAMS,
        )

    if vector_store_type == "23AI":
        dsn = f"{DB_HOST_IP}:1521/{DB_SERVICE}"

        connection = oracledb.connect(user=DB_USER, password=DB_PWD, dsn=dsn)

        v_store = OracleVS(
            client=connection,
            table_name="ORACLE_KNOWLEDGE",
            distance_strategy=DistanceStrategy.COSINE,
            embedding_function=embed_model,
        )

    # 10/05: added qdrant
    if vector_store_type == "QDRANT":
        client = QdrantClient(url=QDRANT_URL)
        collection_name = COLLECTION_NAME
        v_store = Qdrant(client, collection_name, embeddings=embed_model)

    return v_store
