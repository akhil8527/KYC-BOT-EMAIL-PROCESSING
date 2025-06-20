""" 
The redis client for vector store.
"""

import os
import numpy as np
import torch
import redis
from redis.commands.search.query import Query
from redis.commands.search.field import TagField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

from configurations.params import REDIS_STACK, SCORE_THRESHOLD
from source.model import get_model_embed_dim, get_model_name


def redis_embeddings_client() -> redis.Redis:
    """ 
    Get or create the redis vector store.

    Returns:
        the redis index store
    """
    redis_db = redis.Redis(
        host=REDIS_STACK.host,
        port=REDIS_STACK.port,
        password=os.environ["REDIS_HOST_PASSWORD"],
    )

    try:
        # check if redis index already exists
        redis_db.ft(REDIS_STACK.index_name).info()
    except:
        schems = (
            TagField("tag"),
            VectorField(
                "vector", 
                "FLAT",
                {
                    "TYPE": "FLOAT32",
                    "DIM": get_model_embed_dim(), # Number of Vector Dimensions
                    "DISTANCE_METRIC": "COSINE",  # Vector Search Distance Metric
                },
            ),
        )

        # Index Definition
        definition = IndexDefinition(
            prefix=REDIS_STACK.doc_prefix,
            index_type=IndexType.HASH,
        )

        # Create Index
        redis_db.ft(REDIS_STACK.index_name).create_index(fields=schema, definition=definition)

    return redis_db


def add_redis(
        embeddings: torch.Tensor,
        email_ids: list[int],
        email_classes: list[str],
        email_txts: list[str | None],
):
    for email_id in email_ids:
        keys = list(redis_embeddings_client().scan_iter(
            """%(doc_prefix)s:%(email_id)i}:*""",
            {"doc_prefix": REDIS_STACK.doc_prefix, "email_id": email_id},
        ))

        if len(keys) > 0:
            raise ValueError(f"{email_id} already exists in db")
    
    pipe = redis_embeddings_client().pipeline()
    for i, embedding in enumerate(embeddings):
        pipe.hset(
            f"{REDIS_STACK.doc_prefix}:{email_ids[i]}:{email_classes[i]}",
            mapping={
                "vector": embedding.numpy().tobytes(),
                "content": email_txts[i],
                "tag": get_model_name()
            },
        )
    
    res = pipe.execute()
    return res


def search_redis(
    query_embedding: torch.Tensor | np.ndarray,
    top_k: int = 5,
    score_threshold: float = SCORE_THRESHOLD,
):
    """
    Get most similar emails from the redis db

    Args:
        query_embedding: the query vector
        top_k: number of top matching elements to be returned
    Returns:
        top_k number of matching documents
    """

    if isinstance(query_embedding, torch.Tensor):
        query_embedding = query_embedding.numpy()

    query = (
        Query(f"(@tag:{get_model_name()})=>[KNN {top_k} @vector $vec as score]")
        .sort_by("score")
        .return_fields("content", "tag", "score")
        .paging(0, 2)
        .dialect(2)
    )

    query_params = {"vec": query_embedding.tobytes()}
    results = ( 
        redis_embeddings_client()
        .ft(REDIS_STACK.index_name)
        .search(query, query_params)
        .docs
    )

    return results
    

def remove_redis_records(email_ids: list[int]) -> list[bool]:
    """ 
    Remove given class record from the redis

    Args:
        email_id: email_id to be deleted

    Returns:

    """
    for email_id in email_ids:
        for key in redis_embeddings_client().scan_iter(
            """%(doc_prefix)s:%(email_id)i}:*""",
            {"doc_prefix": REDIS_STACK.doc_prefix, "email_id": email_id},
        ):
            redis_embeddings_client().delete(key)