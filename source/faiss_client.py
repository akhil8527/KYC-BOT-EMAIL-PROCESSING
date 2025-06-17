import os
import logging
import faiss

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from configurations.params import FAISS_DB_PATH, LOGGING_FILE_PATH, SCORE_THRESHOLD
from source.model import embedding_model, get_model_embed_dim


FAISS_INDEX: dict[str, FAISS] = {}
""" The faiss index vector store """

REDIS_EMAIL_STORE = None
""" Redis for storing email_id: faiss_loc | class"""

logger = logging.getLogger(__name__)
logging.basicConfig(filename=LOGGING_FILE_PATH, level=logging.INFO)


def faiss_save_all():
    """ Save all open databases """

    for key, value in FAISS_INDEX.items():
        logger.info("Saving faiss vector store to %s", key)
        value.save_local(folder_path=key)


def faiss_index(user: str, faiss_index_path: str = FAISS_DB_PATH) -> FAISS:
    """ 
    Get or Create a new faiss index vector store 
    
    Args:
        user: the user id for whom the vector store should be obtained
        faiss_index_path: the path to the faiss db to be loaded

    Returns:
        the faiss index store
    """
    faiss_index_path = os.path.join(faiss_index_path, user)

    global FAISS_INDEX
    
    if faiss_index_path not in FAISS_INDEX:
        if os.path.exists(faiss_index_path) and os.listdir(faiss_index_path):
            logger.info("FAISS vector store loaded from %s", faiss_index_path)
            FAISS_INDEX[faiss_index_path] = FAISS.load_local(
                faiss_index_path,
                embedding_model,
                allow_dangerous_deserialization=True,
            )
        else:
            logger.info("creating a new FAISS vector store %s", faiss_index_path)
            if not os.path.exists(faiss_index_path):
                os.mkdir(path=faiss_index_path)
            index = faiss.IndexFlatL2(get_model_embed_dim())
            FAISS_INDEX[faiss_index_path] = FAISS(
                embedding_function=embedding_model,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )
    
    return FAISS_INDEX[faiss_index_path]


def add_faiss(
        texts: dict[str, str],
        email_classes: list[str],
        user: str,
        faiss_index_path: str = FAISS_DB_PATH,
        save: bool = False,
) -> list[bool]:
    """ 
    Add new email to the faiss database.

    Args:
        texts: email_id, email_text to be added
        email_classes: corresponding email classes to be assigned
        user: the user id for whom the class labels should be added
        faiss_index_path: the main faiss index path
        save: whether to save the vector store after adding the vector
    
    Returns:
        bool results for each successful/unsuccessful email addition
    """

    process_email_ids = []
    email_ids = texts.keys()
    for email_id in email_ids:
        filtered_keys = [
            key for key in faiss_index(
                user=user,
                faiss_index_path=faiss_index_path,
            ).docstore._dict.keys()
            if key.split(":")[0] == email_id
        ]
    
    if len(filtered_keys) > 0:
        process_email_ids.append(False)
    else:
        process_email_ids.append(True)

    
    email_ids = [
        f"{email_id}:{email_cls}"
        for email_id, email_cls, not_exists in zip(email_ids, email_classes, process_email_ids) if not_exists
    ]

    documents = [
        Document(page_content=text, metadata={"cls": email_cls})
        for text, email_cls, not_exists in zip(texts.values, email_classes, process_email_ids) if not_exists
    ]

    if len(documents)==0:
        return process_email_ids
    
    faiss_index(user=user,
                faiss_index_path=faiss_index_path).add_documents(documents=documents, ids=email_ids)

    if save:
        faiss_index(user=user,
                    faiss_index_path=faiss_index_path).save_local(FAISS_DB_PATH)

    return process_email_ids


def search_faiss(
        query_text: str | list[float],
        user: str,
        top_k: int = 5,
        score_threshold: float | None = SCORE_THRESHOLD,
        faiss_index_path: str = FAISS_DB_PATH,
) -> list[tuple[Document | None, float]]:
    """
    Get most similar emails from faiss db.

    Args: 
        query_text: query_text or query_embedding
        user: user id for whom the search to be performed
        top_k: number of top matching elements to be returned
        score_threshold: the threshold between class-none predictions
        faiss_index_path: the main path to db
    Returns:
        top_k number of matching documents and its scores
    """
    if isinstance(query_text, str):
        results = faiss_index(
            user=user,
            faiss_index_path=faiss_index_path,
        ).similarity_search_with_score(query_text, k=top_k)

    elif (isinstance(query_text, list) and len(query_text)>0 and isinstance(query_text[0], float)):
        results = faiss_index(user=user,
                              faiss_index_path=faiss_index_path).similarity_search_with_score_by_vector(embedding=query_text)
    
    else:
        raise ValueError(f"Unsupported query type {type(query_text)}")
    
    if score_threshold is None:
        score_threshold = 0.0

    filtered_results = [(doc, 1-score) for doc, score in results if (1-score) > score_threshold]

    if len(filtered_results)==0:
        if len(results)==0:
            return None, 1.0
        
        min_score = min([score for _, score in results])

        return None, min_score.item()
    
    return filtered_results


def remove_faiss_records(
        user: str, 
        email_ids: list[str],
        faiss_index_path: str,
) -> bool:
    """ 
    Remove given class record from the redis

    Args:
        user: user id for whom the remove operation should be performed
        email_id: email_id to be be deleted
        faiss_index_path: the faiss index path to remove records from
    Returns:
        True if successful
    """
    if len(email_ids)==0:
        return True
    return faiss_index(user=user,
                       faiss_index_path=faiss_index_path).delete(ids=email_ids)


def get_all_faiss_class_elems(
        cls: list[str],
        user: str,
        faiss_index_path: str
) -> list[str]:
    filtered_keys = [key 
                     for key in faiss_index(user=user, 
                                            faiss_index_path=faiss_index_path).docstore._dict.keys() 
                                            if key.split(":")[1] in cls]
    return filtered_keys


def list_faiss_classes(
        user: str,
        faiss_index_path: str
) -> dict[str, int]:
    """ 
    List all classes present in the faiss db.

    Args:
        faiss_index_path: faiss_path
        user: user id for whom the remove operation should be performed
    Returns:
        class names and records count for each class
    """
    email_classes = {}
    for key in faiss_index(user=user, faiss_index_path=faiss_index_path).docstore._dict.keys():
        email_class = key.split(":")[-1]
        email_id = (":").join(key.split(":")[:-1])
        if email_class in email_classes:
            email_classes[email_class]["total"] += 1
            email_classes[email_class]["email_ids"].append(email_id)
        else:
            email_classes[email_class] = {"total": 1, "email_ids": [email_id]}

    return email_classes