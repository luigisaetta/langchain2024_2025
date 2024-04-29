"""
Author: Luigi Saetta
Date created: 2024-04-27
Date last modified: 2024-04-27
Python Version: 3.11
"""

from tqdm.auto import tqdm
from langchain_community.embeddings import OCIGenAIEmbeddings

from config import EMBED_BATCH_SIZE


#
# extend OCIGenAIEmbeddings adding batching
#
class OCIGenAIEmbeddingsWithBatch(OCIGenAIEmbeddings):
    """
    add batching to OCIEmebeddings
    with Cohere max # of texts is: 96
    """

    def embed_documents(self, texts):
        batch_size = EMBED_BATCH_SIZE
        embeddings = []

        if len(texts) > batch_size:
            # do in batch
            for i in tqdm(range(0, len(texts), batch_size)):
                batch = texts[i : i + batch_size]

                embeddings_batch = super().embed_documents(batch)

                # add to the final list
                embeddings.extend(embeddings_batch)
        else:
            # this way we don't display progress bar when we embed a query
            embeddings = super().embed_documents(texts)

        return embeddings
