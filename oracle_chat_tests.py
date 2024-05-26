"""
Author: Luigi Saetta
Date created: 2024-04-27
Date last modified: 2024-04-27
Python Version: 3.11
"""

from factory import build_rag_chain
from utils import enable_tracing

from config import ENABLE_TRACING, FAISS_DIR, BOOKS_DIR


def answer(question, v_chat_history):
    """
    answer the question
    """
    ai_msg = rag_chain.invoke({"input": question, "chat_history": v_chat_history})
    print(ai_msg["answer"])

    return ai_msg["answer"]


#
# Main
#
# la directory che contiene i pdf
if ENABLE_TRACING:
    enable_tracing()

#
# Inizializza l'intera catena
#
rag_chain = build_rag_chain(FAISS_DIR, BOOKS_DIR, verbose=True)

chat_history = []

# do a test
QUESTION = "Quali sono i farmaci che si possono usare per curare il diabete di tipo 2?"
print(QUESTION)
print("")

# mostra come fare lo streaming
for chunk in rag_chain.stream({"input": QUESTION, "chat_history": chat_history}):
    if "answer" in chunk:
        print(chunk["answer"], end="", flush=True)

# response = answer(question, chat_history)
# print("")

# chat_history.extend([HumanMessage(content=question), response])

# question = "Si può usare la metformina?"
# print(question)
# response = answer(question, chat_history)
# print("")
