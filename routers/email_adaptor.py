"""
Email processing routes definitions 
"""

import platform
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi_pagination import Page
from fastapi_pagination.iterables import paginate
from loguru import logger
from pydantic import BaseModel, field_validator

from configurations.params import FAISS_DB_PATH
from routers.auth import get_current_user
from source.commons.postgress_communication import (
    add_user_classes,
    get_user_classes,
    remove_classes,
)
from source.email_cls_service import (
    Emails,
    add_new_record_faiss,
    dataset_paginate_func,
    email_dataset,
    predict_cls_by_id,
)
from source.faiss_client import (
    get_all_faiss_class_elems,
    list_faiss_classes,
    remove_faiss_records,
)


###########################################
logger.add(
    "./logs/email.log", format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}"
)

uname = platform.uname()

################### App Routers #######################

email_classification = APIRouter(
    tags=["email"],
    prefix="/email"
)

class NewEmails(BaseModel):
    """ Holder for email request data """
    email_ids: list[str]
    """ Email text """

    email_classes: list[str]
    """ The classes to assign to emails """

    def model_post_init(self, __context):
        assert len(self.email_ids) == len(self.email_classes), f"The length of email_ids and email_classes should be equal. Got {len(self.email_ids)} and {len(self.email_classes)}"


class EmailIds(BaseModel):
    """ Holder for email request data """

    email_ids: list[str]
    """ Email text """


class EmailClasses(BaseModel):
    """ Holder for email request data """

    email_classes: list[str]
    """ The classes to assign to emails """


@email_classification.get(
    "/health",
    status_code=status.HTTP_200_OK,
    description="Status report of api and system"
)
async def health():
    return {
        "server": uname.node,
        "service_name": "Sec gov api",
        "status": "alive",
    }

@email_classification.post(
    "/classify",
    status_code=status.HTTP_200_OK,
    description="Classify document into one of pre-defined classes",
)
async def email_classify(
    user: Annotated[dict, Depends(get_current_user)], 
    email_ids: EmailIds,
):
    """
    Classify emails

    Args:
        email_ids (EmailIds): email_ids to be classified
    
    Returns:
        {email_id: class value}
    """
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed."
        )
    
    response = predict_cls_by_id(email_ids=email_ids.email_ids, user=user["user"])
    response = jsonable_encoder(response)

    return JSONResponse(response)


@email_classification.post(
    "/assign_class",
    status_code=status.HTTP_200_OK,
    description="Assigned a class to the email",
)
async def assign_class(
    user: Annotated[dict, Depends(get_current_user)], 
    emails: NewEmails
):
    """ 
    Assign a new or existing class to the given email

    Args:
        email_id: email id to assign the class to
        email_cls: the class to be assigned
    """
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed."
        )

    user_defined_classes = get_user_classes(user_id=user["user"])
    for email_cls in emails.email_classes:
        if email_cls not in user_defined_classes:
            return JSONResponse(
                f"Please define the {email_cls} class first, before using it. The class can be defined using /add_class endpoint"
            )

    res = add_new_record_faiss(
        email_ids=emails.email_ids,
        email_classes=emails.email_classes,
        user=user["user"]
    )

    response = {
        email_id: "already exists in db" if not added else "success" for email_id, added in zip(emails.email_ids, res)
    }

    return JSONResponse(response)


@email_classification.post(
    "/add_class",
    status_code=status.HTTP_200_OK,
    description="Define new email clas",
)
async def add_class(
    user: Annotated[dict, Depends(get_current_user)], 
    classes: list[str],
):
    """ 
    Add user-defined class.
    User is required to create a class label prior to assigning this label to the email

    Args:
        classes: email classes to create

    """
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed."
        )

    results = add_user_classes(user_id=user["user"], classes=list(set(classes)))

    return JSONResponse(results)


@email_classification.post(
    "/remove_whole_class",
    status_code=status.HTTP_200_OK,
    description="Remove class (ulabel all emails with a given class)"
)
async def remove_whole_class(
    user: Annotated[dict, Depends(get_current_user)], 
    email_classes: EmailClasses
):
    """ 
    Remove all records labeled with a given class label.

    Args:
        email_cls: the class to be removed.
    """
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed."
        )
    
    email_ids_to_remove = get_all_faiss_class_elems(
        cls=email_classes.email_classes,
        faiss_index_path=FAISS_DB_PATH,
        user=user["user"],
    )

    response = remove_faiss_records(
        user=user["user"],
        email_ids=email_ids_to_remove,
        faiss_index_path=FAISS_DB_PATH,
    )

    return JSONResponse(response)


@email_classification.get(
    "/get_emails",
    status_code=status.HTTP_200_OK,
    description="Fetch emails",
)
async def get_emails(user: Annotated[dict, Depends(get_current_user)]) -> Page[Emails]:
    """ 
    Fetch emails from storage
    """
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed."
        )
    
    return paginate(dataset_paginate_func(), total=len(email_dataset))


@email_classification.post(
    "/remove_class_record",
    status_code=status.HTTP_200_OK,
    description="Unlabel an email",
)
async def remove_class_record(
    user: Annotated[dict, Depends(get_current_user)], 
    email_ids: list[int]
):
    """
    Unlabel an email

    Args:
        email_ids: email ids to unlabel
    """
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed."
        )
    
    response = remove_faiss_records(
        user=user["user"],
        email_ids=email_ids,
    )

    return JSONResponse(response)


@email_classification.post(
    "/get_available_classes",
    status_code=status.HTTP_200_OK,
    description="Get available email classes",
)
async def get_available_classes(user: Annotated[dict, Depends(get_current_user)]):
    """ 
    Get all available email classes

    Returns:
        list of available email classes
    """
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed."
        )
    
    user_defined_classes = get_user_classes(user_id=user["user"])

    return JSONResponse(user_defined_classes)


@email_classification.get(
    "/get_assigned_classes",
    status_code=status.HTTP_200_OK,
    description="Get assigned email classes."
)
async def get_assigned_classes(user: Annotated[dict, Depends(get_current_user)]):
    """ 
    Get all assigned email classes

    Returns:
        list of assigned email classes
    """
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed."
        )
    
    response = list_faiss_classes(faiss_index_path=FAISS_DB_PATH, user=user["user"])

    return JSONResponse(response)


@email_classification.post(
    "/remove_class",
    status_code=status.HTTP_200_OK,
    description="Remove class from the set of user defined classes",
)
async def remove_class(
    user: Annotated[dict, Depends(get_current_user)], 
    email_classes: list[str]
):
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed."
        )
    
    response = remove_classes(
        user_id=user["user"],
        classes=email_classes
    )

    email_ids_to_remove = get_all_faiss_class_elems(
        cls=[key for key, value in response.items() if value],
        faiss_index_path=FAISS_DB_PATH,
        user=user["user"],
    )

    response = remove_faiss_records(
        user=user["user"],
        email_ids=email_ids_to_remove,
        faiss_index_path=FAISS_DB_PATH,
    )

    return JSONResponse(response)