"""
Author: Luigi Saetta
Date created: 2024-04-27
Date last modified: 2024-04-27
Python Version: 3.11
"""

from langchain_core.prompts import ChatPromptTemplate

#
# this is the prompt for the final answer
#
prompt_4_answer = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You're an AI assistant. 
            Given a user question and some documents, answer the user question. 
            If none of the articles answer the question, just say you don't know.
            
            Here are the documents:
            {context}""",
        ),
        ("human", "{question}"),
    ]
)
