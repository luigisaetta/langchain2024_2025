"""
Author: Luigi Saetta
Date created: 2024-04-27
Date last modified: 2024-04-28
Python Version: 3.11

Usage: this module can be used to rebuild the FAISS index, for example when adding new books
"""

from chunk_index_utils import load_and_rebuild_faiss_index
from factory import get_embed_model
from config import FAISS_DIR, BOOKS_DIR


#
# Reload and save the Vector Store
# can be used to add more books
#

#
# Main
#
embed_model = get_embed_model()

load_and_rebuild_faiss_index(FAISS_DIR, BOOKS_DIR, embed_model)
