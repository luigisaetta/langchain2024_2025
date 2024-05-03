"""
Author: Luigi Saetta
Date created: 2024-04-27
Date last modified: 2024-04-27
Python Version: 3.11
"""

import logging
import os

from config import (
    TOP_K,
    TOP_N,
    EMBED_MODEL_TYPE,
    OCI_EMBED_MODEL,
    COHERE_EMBED_MODEL,
    ENABLE_TRACING,
    LANGCHAIN_PROJECT,
    ADD_RERANKER,
    COHERE_RERANKER_MODEL,
    VECTOR_STORE_TYPE,
    LLM_MODEL_TYPE,
    COHERE_GENAI_MODEL,
)
from config_private import LANGSMITH_API_KEY


def remove_path_from_ref(ref_pathname):
    """
    remove the path from source (ref)
    """
    ref = ref_pathname
    if len(ref_pathname.split(os.sep)) > 0:
        ref = ref_pathname.split(os.sep)[-1]

    return ref


def load_configuration():
    """
    read the configuration from config and return a configs dictionary
    """
    configs = {}
    configs["VECTOR_STORE_TYPE"] = VECTOR_STORE_TYPE
    configs["EMBED_MODEL_TYPE"] = EMBED_MODEL_TYPE
    configs["OCI_EMBED_MODEL"] = OCI_EMBED_MODEL
    configs["COHERE_EMBED_MODEL"] = COHERE_EMBED_MODEL
    configs["LLM_MODEL_TYPE"] = LLM_MODEL_TYPE
    configs["TOP_K"] = TOP_K
    configs["TOP_N"] = TOP_N

    return configs


def enable_tracing():
    """
    To enable tracing with LangSmith
    """
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY


def get_console_logger():
    """
    To get a logger to print on console
    """
    logger = logging.getLogger("ConsoleLogger")

    # to avoid duplication of logging
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False

    return logger


def format_docs(docs):
    """
    format docs for LCEL
    """
    return "\n\n".join(doc.page_content for doc in docs)


def print_configuration():
    """
    print the current config
    """
    logger = logging.getLogger("ConsoleLogger")

    logger.info("--------------------------------------------------")
    logger.info("Configuration used:")
    logger.info("")

    logger.info(" Embedding model type: %s", EMBED_MODEL_TYPE)

    if EMBED_MODEL_TYPE == "OCI":
        logger.info(" Using %s for Embeddings...", OCI_EMBED_MODEL)
    if EMBED_MODEL_TYPE == "COHERE":
        logger.info(" Using %s for Embeddings...", COHERE_EMBED_MODEL)

    if ADD_RERANKER:
        logger.info(" Added Cohere Reranker...")
        logger.info(" Using %s as reranker...", COHERE_RERANKER_MODEL)

    logger.info(" Using %s as Vector Store...", VECTOR_STORE_TYPE)
    logger.info(" Retrieval parameters:")
    logger.info("    TOP_K: %s", TOP_K)
    logger.info("    TOP_N: %s", TOP_N)

    logger.info(
        " Using %s, %s as Generative AI Model...", LLM_MODEL_TYPE, COHERE_GENAI_MODEL
    )
    if ENABLE_TRACING:
        logger.info("")
        logger.info(" Enabled Observability with LangSmith...")

    logger.info("--------------------------------------------------")
    logger.info("")


def check_value_in_list(value, values_list):
    """
    to check that we don't enter a not supported value
    """
    if value not in values_list:
        raise ValueError(
            f"Value {value} is not valid: value must be in list {values_list}"
        )


def answer(chain, question):
    """
    method to test answer
    """
    response = chain.invoke({"question": question})

    print(question)
    print("")
    print(response["answer"])
    print("")
