"""
Email classification service code
"""

import re
from collections import Counter
from typing import Iterable

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from configurations.params import DATASET_PATH, FAISS_DB_PATH, SCORE_THRESHOLD
from source.dataset import EmailDataset
from source.faiss_client import add_faiss, search_faiss
from source.model import embedding_model
from source.redis_stack_client import add_redis, search_redis


class Emails(BaseModel):
    """ Holder for email data """

    email_id: str = Field(example="change_request1")
    file_name: str = Field(example="")

email_dataset = EmailDataset(dataset_path=DATASET_PATH, cache=True)


def fetch_email_by_ids(email_ids: list[int]) -> dict[int, str]:
    
    emails_dict = {}
    classes_counter = {}

    for email_id in email_ids:
        res = re.match(".*?([0-9]+)$", email_id)
        assert res is not None
        email_cls = email_id[:res.regs[1][0]]
        email_num = email_id[res.regs[1][0]:]
        assert email_num.isnumeric()
        assert email_cls in list(email_dataset.dict_data.keys())
        assert int(email_num) < len(email_dataset.dict_data[email_cls])

        if email_cls not in classes_counter:
            classes_counter[email_cls] = 1
        else:
            classes_counter[email_cls] += 1

        email_elem = email_dataset.dict_data[email_cls][int(email_num)]
        assert email_id == email_elem.email_id, f"{email_id} != {email_elem.email_id}"
        emails_dict[email_id] = email_elem.email_text

    return emails_dict


def predict_cls_by_id(email_ids: str | list[str], user: str) -> dict[str, dict[str, float | str]]:
    """
    Predict the email class given the email ids.

    Args:
        email_ids (str | list[str]): ids to make predictions on
        user (str): User id
    """
    if isinstance(email_ids, int):
        email_ids = [email_ids]
    
    email_text = fetch_email_by_ids(email_ids=email_ids)

    predictions = predict_cls_by_text_faiss(emails_text=email_text, user=user)

    return predictions


def majority_score(predicted_classes: list[tuple[Document, float]]) -> tuple[str, float]:
    """ 
    Compute the class label as a majority score
    """
    predictions = []
    scores = []

    for pred, score in predicted_classes:
        predictions.append(pred)
        scores.append(score)

    counter = Counter([doc.id.split(":")[-1] for doc in predictions])

    classes = {}
    classes_counter = {}
    for doc, score in zip(predictions, score):
        pred_cls = doc.id.split(":")[-1]

        if pred_cls not in classes:
            classes[pred_cls] = score
            classes_counter[pred_cls] = 1
        else:
            classes[pred_cls] += score
            classes_counter[pred_cls] += 1

    for key, value in classes.items():
        classes[key] = value / classes_counter[key]

    best_pred = [(value, count, classes[value]) for value, count in counter.most_common()]

    best_pred.sort(
        key=lambda item: (item[1], item[2]), reverse=True
    ) # first sort by number of elements, then by score
    
    value, count, score = best_pred[0]
    
    return value, score


def predict_cls_by_text_faiss(
    emails_text: dict[str, str],
    user: str,
    faiss_index_path: str = FAISS_DB_PATH,
    threshold: float | None = SCORE_THRESHOLD,
) -> dict[str, dict[str, float | str]]:
    """
    Predict the email class label given email text and set of user defined classes.

    Args:
        email_text: text to be classified

    Returns:
        predicted cls and confidence
    """
    predicted_classes = {}
    for email_id, email_text in emails_text.items():
        top_matches = search_faiss(
            query_text=email_text,
            user=user,
            score_threshold=threshold,
            faiss_index_path=faiss_index_path,
        )

        if len(top_matches) == 0:
            predicted_classes[email_id] = "None"
            continue
        
        if top_matches[0] is None:
            predicted_classes[email_id] = {
                "pred_class": "None",
                "conf": top_matches[1],
            }
            continue
        
        pred_cls, conf = majority_score(predicted_classes=top_matches)
        predicted_classes[email_id] = {
            "pred_class": pred_cls,
            "conf": conf,
        }

        return predicted_classes
    

def predict_cls_by_text_redis(
    emails_text: dict[str, str],
    classes: set[str],
) -> tuple[str, float]:
    """ 
    Predict the email class label given email text and set of user defined classes

    Args:
        email_text: text to be classified
        classes: classes to be used

    Returns:
        predicted cls and confidence
    """
    email_embeddings = embedding_model().encode(list(emails_text.values()))
    predicted_classes = {}
    for email_id, email_embedding in zip(emails_text.keys(), email_embeddings):
        top_matches = search_redis(query_embedding=email_embedding)
        pred_cls = majority_score(predicted_classes=top_matches)
        predicted_classes[email_id] = pred_cls

    return pred_cls


def add_new_record_redis(
        email_ids: list[int],
        email_classes: list[str],
):
    assert len(email_ids) == len(email_classes), f"The length of email_ids and email_classes should be equal. Got {len(email_ids)} and {len(email_classes)}"
    
    emails_text = fetch_email_by_ids(email_ids=email_ids)
    email_embeddings = embedding_model().encode(list(emails_text.values()))

    res = add_redis(embeddings=email_embeddings, 
                    email_ids=email_ids,
                    email_classes=email_classes)
    
    return res


def add_new_record_faiss(
        email_ids: list[int],
        email_classes: list[str],
        user: str
) -> list[bool]:
    """ 
    Add new class groundtruth to the faiss vector store

    Args:
        email_ids: email_ids for which to assign class
        email_classes: corresponding email classes to be assigned
        user: user id for which to create new records
    """

    assert len(email_ids) == len(email_classes), f"""The length of email_ids and email_classes should be equal. Got {len(email_ids)} and {len(email_classes)}"""
    
    emails_text = fetch_email_by_ids(email_ids=email_ids)
    res = add_faiss(texts=emails_text, email_classes=email_classes, user=user)
    return res


def paginate_func() -> Iterable[Emails]:
    for elem in email_dataset:
        yield Emails(email_id=elem.email_id, file_name=elem.email_filename)