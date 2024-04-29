"""
Author: Luigi Saetta
Date created: 2024-04-27
Date last modified: 2024-04-27
Python Version: 3.11
"""

import sys
import logging
import argparse
import numpy as np

from tqdm.auto import tqdm

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

from tokenizers import Tokenizer
import requests

from utils import get_console_logger
from config import CHUNK_OVERLAP, BOOKS_DIR

#
# Main
#
logger = get_console_logger()

parser = argparse.ArgumentParser(description="Analyze chunking.")
parser.add_argument("max_chunk_size", type=int, help="Max. dim. of a chunk in chars")

args = parser.parse_args()

if args.max_chunk_size is not None:
    logging.info("Running with : %s max_chunk_size...", args.max_chunk_size)
    max_chunk_size = args.max_chunk_size
else:
    logging.error("No params provided !!!")
    sys.exit(-1)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=max_chunk_size,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    is_separator_regex=False,
)

loader = PyPDFDirectoryLoader(path=BOOKS_DIR, glob="*.pdf")

logger.info("")
logger.info("Loading and splitting in chunks...")

docs = loader.load_and_split(text_splitter=text_splitter)

logger.info("max_chunk_size: %s chars...", max_chunk_size)
logger.info("Loaded %s chunks...", len(docs))
logger.info("")
logger.info("Analyzing chunks...")
logger.info("")

# load the Cohere tokenizer
# here define the tokenizer to use (linked to the embedding model)
MODEL_NAME = "embed-multilingual-v3"

tokenizer_url = (
    f"https://storage.googleapis.com/cohere-assets/tokenizers/{MODEL_NAME}.json"
)

response = requests.get(tokenizer_url, timeout=60)
tokenizer = Tokenizer.from_str(response.text)

# max num. of tokens for Cohere embeddings input
THRESHOLD = 512

tokens_list = []

# here we compute the num. of tokens per chunk using Cohere tokenizer
for doc in tqdm(docs):
    n_tokens = len(
        tokenizer.encode(sequence=doc.page_content, add_special_tokens=False)
    )
    tokens_list.append(n_tokens)

logger.info("Results:")

np_toks = np.array(tokens_list)

# to compute how many are longer than threshold
mask = np_toks > THRESHOLD

logger.info("")
logger.info("Avg. # of tokens per chunk: %s", round(np.mean(np_toks)))

max_toks = np.max(np_toks)
min_toks = np.min(np_toks)
perc75 = round(np.percentile(np_toks, 75))
logger.info("Max: %s, Min: %s, 75-perc.: %s tokens", max_toks, min_toks, perc75)

sum_longer = np.sum(mask)
perc = round(np.sum(mask) * 100.0 / len(tokens_list), 1)
logger.info(
    "Num. of chunks longer than %s tokens: %s (%s perc.)", THRESHOLD, sum_longer, perc
)
logger.info("")
