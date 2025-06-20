import os
import re
import time
import uuid
import logging
import datetime
from typing import Annotated

import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from ldap3 import SUBTREE, Connection, Server
from pydantic import BaseModel

from configurations.config_reader import read_config

secure = APIRouter(
    tags=["auth"],
    prefix="/auth"
)

config = read_config("config.ini")
ENV = os.getenv("ENV_TYPE", "DEV")
ENV_DICT = {"D": "DEV", "U": "UAT", "P": "PROD"}

oauth2_bearer = OAuth2PasswordBearer(tokenUrl="auth/get_token")

class Token(BaseModel):
    status: str
    access_token: str

async def get_current_user(token: Annotated[str, Depends(oauth2_bearer)]):
    try:
        payload = jwt.decode(
            token, 
            config[ENV]["secret_key"], 
            algorithms=[config[ENV]["algorithm"]]
        )
        return payload
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
        )
    

@secure.post("/get_token", 
             status_code=status.HTTP_201_CREATED, 
             response_model=Token, 
             description="Get authentication token")
async def generate_auth_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> dict[str, str]:
    """ 
    Generate a new authentication token

    Args:
        form_data: user credentials
    
    Returns:
        access token or error
    """
    try: 
        # data to be signed using token
        identifier = str(uuid.uuid4() + str(time.time()))
        is_authenticated, ldap_conn = authenticate_user(
            form_data.username,
            form_data.password
        )
        user_app_access_dict_list = get_user_access_info(ldap_conn, form_data.username)
        user_info = {
            "user": form_data.username,
            "exp": datetime.datetime.utcnow() + datetime.timedelta(seconds=90),
            "id": identifier,
            "user_app_access_dict_list": user_app_access_dict_list,
        }
        
        try:
            token = jwt.encode(
                user_info, 
                config[ENV]["secret_key"],
                algorithm=config[ENV]["algorithm"]
            )
            return {"status": "success", "access_token": token}
        
        except jwt.ExpiredSignatureError:
            return {"status": "error", "access_token": "Token expired"}
        
        except jwt.InvalidTokenError:
            return {"status": "error", "access_token": "Invalid token"}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
        )
    

def authenticate_user(username: str, password: str) -> tuple[bool, Connection]:
    """ 
    Authenticate user, using username and password

    Args:
        username: username of the user
        password: password of the user

    Returns:
        True if authentication successful, False otherwise
    """
    is_authenticated = False
    conn = None
    print("in authenticate user")
    print("config[ENV][ldap_server]: ", config[ENV]["ldap_server"])

    try:
        conn = Connection(
            server=Server(config[ENV]["ldap_server"]),
            user=f"CN={username},OU=HSBCPeople,DC=InfoDir,DC=Prod,DC=HSBC",
            password=password,
            auto_bind=True,
        )
        is_authenticated=True
        logging.info("User is authenticated...")

    except Exception as e:
        conn = Connection(
            server=Server(config[ENV]["ldap_server"]),
            user=f"CN={username},OU=Alternate Accounts,OU=HSBCPeople,DC=InfoDir,DC=Prod,DC=HSBC",
            password=password,
            auto_bind=True,
        )
        is_authenticated=True
        logging.info("User is authenticated service account...")

    return is_authenticated, conn


def get_user_access_info(ldap_conn: Connection, username: str) -> list:
    """ 
    Searching groups to which user is assigned

    Args:
        ldap_conn: ldap connection
        username: username of the user

    Returns:
        list of groups to which user is assigned
    """

    try:
        search_base = "OU=HSBCPeople,DC=InfoDir,DC=Prod,DC=HSBC"
        is_search_success = ldap_conn.search(
            search_base=search_base,
            search_scope=SUBTREE,
            search_filter=f"(&(|(objectclass=userproxy)(objectclass=user))(hsbc-ad-SAMAccountName={username}))",
            attributes=["memberOf"],
        )

        if is_search_success:
            print("Search successful, Found the records.")
            member_of_group_list = ldap_conn.response[0]["attributes"]["memberOf"]

    except Exception as e:
        logging.exception(e)

    
    assigned_group_list=[]
    for group_info_string in member_of_group_list:
        for matched in re.finditer(r"CN=([\w\W]*)", group_info_string.split(",")[0], re.IGNORECASE):
            if matched is not None:
                start_index = matched.span(1)[0]
                end_index = matched.span(1)[1]
                assigned_group_list.append(group_info_string[start_index: end_index])
                
    # after extracting group names, we will only use group with DCREST type
    user_app_access_dict_list = []
    for group_val in assigned_group_list:
        if "DCREST" in group_val:
            group_val_splited = group_val.split("-")  # ["Infodir", "DCREST", "DCRESTOCRP", "ADMIN"]
            core_string = group_val_splited[2]
            role_type = group_val_splited[3]

            if "DCREST" in core_string:
                core_string = core_string.replace("DCREST", "") # core string contains DCREST value which is not requires for identification

                env_type = ENV_DICT[core_string[-1]] # last letter of core string DCREST value which is authorized to use the app

                sub_app_name = core_string[0:-1]

                user_app_access_dict = {
                    "app_name": sub_app_name,
                    "env": env_type,
                    "role": role_type,
                    "group": group_val
                }
                
                user_app_access_dict_list.append(user_app_access_dict)
    
    if len(user_app_access_dict_list) < 1 or len(member_of_group_list) < 1:
        raise Exception("User is not authorized for the DCREST group")
    
    return user_app_access_dict_list