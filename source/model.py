import os
import json
from functools import cache

import torch
import torch.nn.functional as F
from langchain_core.embeddings import Embeddings
from transformers import AutoModel, AutoTokenizer

from configurations.params import EMBEDDING_MODEL_CHECKPOINT

EMBEDDING_MODEL: torch.nn.Module = None
"""The model for text embeddings creation."""


class EmbeddingsModel(Embeddings):

    def __init__(self, model_checkpoint_path: str = EMBEDDING_MODEL_CHECKPOINT):
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_path)
        self.model = AutoModel.from_pretrained(model_checkpoint_path).eval()

    def average_pool(self, 
                     last_hidden_states: torch.Tensor, 
                     attention_mask: torch.Tensor) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )

        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    def grad_pass(self, input_text: list[str] | str) -> torch.Tensor:
        if isinstance(input_text, str):
            input_text = [input_text]

        batch_dict = self.tokenizer(
            input_text,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        outputs = self.model(**batch_dict)
        embeddings = self.average_pool(
            last_hidden_states = outputs.last_hidden_state,
            attention_mask = batch_dict["attention_mask"],
        )

        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings
    
    @torch.inference_mode()
    def __call__(self, input_text: list[str] | str) -> torch.Tensor:
        embeddings = self.grad_pass(input_text)
        return embeddings
    
    def embed_documents(self, texts: list[str]):
        embed_vector = self(input_text = texts)
        return embed_vector.tolist()
    
    def embed_query(self, text: str):
        embed_vector = self(input_text = [text]).squeeze()
        return embed_vector
    

@cache
def get_model_name(embedding_model_checkpoint_path: str = EMBEDDING_MODEL_CHECKPOINT) -> str:
    with open(os.path.join(embedding_model_checkpoint_path, "config.json"), "r") as jsonfile:
        model_data = json.load(jsonfile)

    model_name = model_data["_name_or_path"]
    assert len(model_name.split()), "Model should have a single word name."

    return model_name


@cache
def get_model_embed_dim(embedding_model_checkpoint_path: str = EMBEDDING_MODEL_CHECKPOINT) -> int:
    with open(os.path.join(embedding_model_checkpoint_path, "config.json"), "r") as jsonfile:
        model_data = json.load(jsonfile)

    embedding_size = model_data["hidden_size"]

    return embedding_size


def embedding_model() -> torch.nn.Module:
    """ 
    Get or create the text embedding model

    Args:
        embedding_model_checkpoint_path: the embedding model checkpoint path

    Returns:
        the embedding model
    """
    global EMBEDDING_MODEL
    if EMBEDDING_MODEL is None:
        EMBEDDING_MODEL = EmbeddingsModel(
            model_checkpoint_path=embedding_model_checkpoint_path
        )

    return EMBEDDING_MODEL