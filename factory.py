"""
Author: Luigi Saetta
Date created: 2024-04-27
Date last modified: 2024-04-28
Python Version: 3.11
"""

import os
import logging

from langchain_community.vectorstores import FAISS

# Cohere
from langchain_cohere import ChatCohere, CohereRerank
from langchain.retrievers import ContextualCompressionRetriever

# to handle conversational memory
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from chunk_index_utils import load_and_rebuild_faiss_index
from oci_cohere_embeddings_utils import OCIGenAIEmbeddingsWithBatch

# prompts
from oracle_chat_prompts import CONTEXT_Q_PROMPT, QA_PROMPT

from utils import print_configuration

from config import (
    EMBED_MODEL,
    ENDPOINT,
    COHERE_GENAI_MODEL,
    TEMPERATURE,
    MAX_TOKENS,
    TOP_K,
    TOP_N,
    ADD_RERANKER,
    COHERE_RERANKER_MODEL,
)
from config_private import COMPARTMENT_ID, COHERE_API_KEY


def get_embed_model():
    """
    get the Embeddings Model
    """
    embed_model = OCIGenAIEmbeddingsWithBatch(
        auth_type="API_KEY",
        model_id=EMBED_MODEL,
        service_endpoint=ENDPOINT,
        compartment_id=COMPARTMENT_ID,
    )
    return embed_model


def get_llm():
    """
    todo
    """
    llm = ChatCohere(
        cohere_api_key=COHERE_API_KEY,
        model=COHERE_GENAI_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    return llm


#
# get the Vector Store
#
def get_vector_store(local_index_dir, books_dir, embed_model):
    """
    todo
    """
    logger = logging.getLogger("ConsoleLogger")

    if os.path.exists(local_index_dir):
        logger.info("Loading Vector Store from local dir %s...", local_index_dir)

        v_store = FAISS.load_local(
            local_index_dir, embed_model, allow_dangerous_deserialization=True
        )
        logger.info("Loaded %s chunks of text !!!", v_store.index.ntotal)
    else:
        load_and_rebuild_faiss_index(local_index_dir, books_dir, embed_model)

    return v_store


#
# create the entire RAG chain
#
def get_rag_chain(local_index_dir, books_dir, verbose):
    """
    Build the entire RAG chain

    index_dir: the directory where the local index is stored
    books_dir: the directory where all pdf to load and chunk are stored
    """
    logger = logging.getLogger("ConsoleLogger")

    # print all the used configuration to the console
    print_configuration()

    embed_model = get_embed_model()

    v_store = get_vector_store(local_index_dir, books_dir, embed_model)

    base_retriever = v_store.as_retriever(k=TOP_K)

    # add the reranker
    if ADD_RERANKER:
        if verbose:
            logger.info("Adding a reranker...")

        cohere_rerank = CohereRerank(
            cohere_api_key=COHERE_API_KEY, top_n=TOP_N, model=COHERE_RERANKER_MODEL
        )

        retriever = ContextualCompressionRetriever(
            base_compressor=cohere_rerank, base_retriever=base_retriever
        )
    else:
        # no reranker
        retriever = base_retriever

    llm = get_llm()

    # 1. create a retriever using chat history
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, CONTEXT_Q_PROMPT
    )

    # 2. create the chain for answering
    # we need to use a different prompt
    question_answer_chain = create_stuff_documents_chain(llm, QA_PROMPT)

    # 3, the entire chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # this returns sources and can be streamed
    return rag_chain
