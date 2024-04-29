"""
File name: oracle_chat_with_memory.py
Author: Luigi Saetta
Date created: 2023-12-04
Date last modified: 2024-04-28
Python Version: 3.11

Description:
    This module provides the UI for the RAG demo 

Usage:
    streamlit run oracle_chat_with_memory.py

License:
    This code is released under the MIT License.

Notes:
    This is part of a  series of demo developed using OCI GenAI and LangChain

Warnings:
    This module is in development, may change in future versions.
"""

import time
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from factory import get_rag_chain
from utils import get_console_logger, enable_tracing

#
# Configs
#
from config import (
    VERBOSE,
    HELLO_MSG,
    ENABLE_TRACING,
    FAISS_DIR,
    BOOKS_DIR,
    TITLE,
    ADD_REFERENCES,
    DO_STREAMING,
)

# Constant
USER = "user"
ASSISTANT = "assistant"


# when push the button reset the chat_history
def reset_conversation():
    """
    when push the button reset the chat_history
    """
    # chat_history is per session
    st.session_state.chat_history = []

    st.session_state.request_count = 0


# defined here to avoid import of streamlit in other module
# cause we need here to use @cache
@st.cache_resource
def create_chat_engine(verbose=VERBOSE):
    """
    Create the entire RAG chain
    """
    return get_rag_chain(FAISS_DIR, BOOKS_DIR, verbose=verbose)


def format_references(docs):
    """
    format the references to add at the end of response
    """
    references = "\n\nReferences:\n\n"
    for doc in docs:
        references += f"- {doc.metadata['source']}, pag: {doc.metadata['page']}\n"
    return references


# case no streaming: to format output with references
def nostream_output(v_ai_msg):
    """
    format the output when not using streaming
    """
    formatted_output = v_ai_msg["answer"]

    if ADD_REFERENCES and v_ai_msg["context"]:
        formatted_output += format_references(v_ai_msg["context"])

    st.markdown(formatted_output)

    return formatted_output


# case streaming
def stream_output(v_ai_msg):
    """
    format the output when using streaming
    """
    text_placeholder = st.empty()
    formatted_output = ""

    for chunk in v_ai_msg:
        if "answer" in chunk:
            formatted_output += chunk["answer"]
            text_placeholder.markdown(formatted_output, unsafe_allow_html=True)

        if ADD_REFERENCES:
            if "context" in chunk:
                refs = format_references(chunk["context"])

    # references must be added at the end
    # in Langchain they're passed before the answer in the stream
    if ADD_REFERENCES:
        formatted_output += refs

    text_placeholder.markdown(formatted_output, unsafe_allow_html=True)

    return formatted_output


def display_msg_on_rerun(chat_hist):
    """
    display all the msgs on rerun
    """
    for msg in chat_hist:
        # transform a msg in a dict
        if isinstance(msg, HumanMessage):
            the_role = USER
        else:
            the_role = ASSISTANT

        message = {"role": the_role, "content": msg.content}

        with st.chat_message(message["role"]):
            st.markdown(message["content"])


#
# Main
#

# Configure logging
# I have changed the way I config logger to solve some problems with
# PY 3.11

logger = get_console_logger()

if ENABLE_TRACING:
    # enable tracing with LangSmith
    enable_tracing()

# the title (from config)
st.title(TITLE)

# Added reset button
st.button("Clear Chat History", on_click=reset_conversation)

# Initialize chat history
if "chat_history" not in st.session_state:
    reset_conversation()

# init RAG
with st.spinner("Initializing RAG chain..."):
    # here we create the query engine
    rag_chain = create_chat_engine(verbose=VERBOSE)


# Display chat messages from history on app rerun
display_msg_on_rerun(st.session_state.chat_history)

#
# Here the code where react to user input
#
if question := st.chat_input(HELLO_MSG):
    # Display user message in chat message container
    st.chat_message(USER).markdown(question)

    # Add user message to chat history
    st.session_state.chat_history.append(HumanMessage(content=question))

    # here we call the RAG chain...
    try:
        with st.spinner("Calling AI..."):
            time_start = time.time()

            st.session_state.request_count += 1
            logger.info("")
            logger.info("Question n. %s", st.session_state.request_count)

            #
            # Here we invoke the GenAI service
            #

            # prepare the input adding chat_history
            input_msg = {
                "input": question,
                "chat_history": st.session_state.chat_history,
            }

            if DO_STREAMING:
                ai_msg = rag_chain.stream(input_msg)
            else:
                ai_msg = rag_chain.invoke(input_msg)

        # Display the response in chat message container
        with st.chat_message(ASSISTANT):
            if DO_STREAMING:
                output = stream_output(ai_msg)
            else:
                output = nostream_output(ai_msg)

            # Add assistant response to chat history
            st.session_state.chat_history.append(AIMessage(content=output))

        logger.info("Elapsed time: %s sec.", round((time.time() - time_start), 1))

    except Exception as e:
        ERR_MSG = "An error occurred: " + str(e)
        logger.error(ERR_MSG)
        st.error(ERR_MSG)
