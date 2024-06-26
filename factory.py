"""
Author: Luigi Saetta
Date created: 2024-04-27
Date last modified: 2024-05-20
Python Version: 3.11
"""

import logging

# Cohere
from langchain_cohere import ChatCohere, CohereRerank, CohereEmbeddings
from langchain.retrievers import ContextualCompressionRetriever

# to handle conversational memory
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# LLM
from langchain_community.llms import OCIGenAI

from factory_vector_store import get_vector_store
from oci_cohere_embeddings_utils import OCIGenAIEmbeddingsWithBatch

# prompts
from oracle_chat_prompts import CONTEXT_Q_PROMPT, QA_PROMPT

from utils import print_configuration, check_value_in_list

from config import (
    EMBED_MODEL_TYPE,
    OCI_EMBED_MODEL,
    COHERE_EMBED_MODEL,
    ENDPOINT,
    VECTOR_STORE_TYPE,
    COHERE_GENAI_MODEL,
    OCI_GENAI_MODEL,
    TEMPERATURE,
    MAX_TOKENS,
    TOP_K,
    TOP_N,
    ADD_RERANKER,
    COHERE_RERANKER_MODEL,
    LLM_MODEL_TYPE,
)

from config_private import (
    COMPARTMENT_ID,
    COHERE_API_KEY,
)


#
# functions
#


def get_embed_model(model_type="OCI"):
    """
    get the Embeddings Model
    """
    check_value_in_list(model_type, ["OCI", "COHERE"])

    if model_type == "OCI":
        embed_model = OCIGenAIEmbeddingsWithBatch(
            auth_type="API_KEY",
            model_id=OCI_EMBED_MODEL,
            service_endpoint=ENDPOINT,
            compartment_id=COMPARTMENT_ID,
        )
    if model_type == "COHERE":
        embed_model = CohereEmbeddings(
            model=COHERE_EMBED_MODEL, cohere_api_key=COHERE_API_KEY
        )
    return embed_model


def get_llm(model_type):
    """
    Build and return the LLM client
    """
    check_value_in_list(model_type, ["OCI", "COHERE"])

    if model_type == "OCI":
        llm = OCIGenAI(
            auth_type="API_KEY",
            model_id=OCI_GENAI_MODEL,
            service_endpoint=ENDPOINT,
            compartment_id=COMPARTMENT_ID,
            model_kwargs={"max_tokens": MAX_TOKENS, "temperature": TEMPERATURE},
        )
    if model_type == "COHERE":
        llm = ChatCohere(
            cohere_api_key=COHERE_API_KEY,
            model=COHERE_GENAI_MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
    return llm


#
# create the entire RAG chain
#
def build_rag_chain(local_index_dir, books_dir, verbose):
    """
    Build the entire RAG chain

    index_dir: the directory where the local index is stored
    books_dir: the directory where all pdf to load and chunk are stored
    """
    logger = logging.getLogger("ConsoleLogger")

    # print all the used configuration to the console
    print_configuration()

    embed_model = get_embed_model(EMBED_MODEL_TYPE)

    v_store = get_vector_store(
        vector_store_type=VECTOR_STORE_TYPE,
        embed_model=embed_model,
        local_index_dir=local_index_dir,
        books_dir=books_dir,
    )

    # 10/05: I can add a filter here (for ex: to filter by profile_id)
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

    llm = get_llm(model_type=LLM_MODEL_TYPE)

    # steps to add chat_history
    # 1. create a retriever using chat history
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, CONTEXT_Q_PROMPT
    )

    # 2. create the chain for answering
    # we need to use a different prompt from the one used to
    # condense the standalone question
    question_answer_chain = create_stuff_documents_chain(llm, QA_PROMPT)

    # 3, the entire chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # this returns sources and can be streamed
    return rag_chain
