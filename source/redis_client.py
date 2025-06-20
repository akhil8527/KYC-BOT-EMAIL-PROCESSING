""" 
The redis client for storing pre-computed email embeddings
"""

import os
import torch
import redis

from configurations.params import REDIS

REDIS_EMBEDDINGS_STORE = None
"""Redis for storing email_id: embedding vector"""

def redis_embeddings_client() -> redis.Redis:
    """
    Get or create the redis vector store

    Returns:
        the redis index store
    """
    redis_db = redis.Redis(
        host=REDIS.host,
        port=REDIS.port,
        password=os.environ["REDIS_HOST_PASSWORD"],
        db=0,
    )

    return redis_db

def add_redis(
        embeddings: torch.Tensor, 
        email_ids: list[int],
    ):
    pipe = redis.embeddings_client().pipeline()
    for email_id, embedding in zip(email_ids, embeddings):
        pipe.set(email_id, embedding)

    res = pipe.execute()
    return res