import os
import logging
import psycopg2

from configurations.params import database, db_host, db_password, db_port, user
from source.commons.crypt_utils import AESCipher

logger = logging.getLogger(__name__)

PASSWORD: str | None = None
""" Password to PostgreSQL """

POSTGRESS_CONNECTION: psycopg2.extensions.connection | None = None
""" Connection to PostgreSQL """

def password() -> str:
    """ Get or decrypt the postgres db password """

    global PASSWORD
    if PASSWORD is None:
        PASSWORD = AESCipher().decrypt(db_password)
    return PASSWORD


def postgress_connection() -> psycopg2.extensions.connection | None:
    """ 
    Get or create connection to PostgreSQL

    Returns:
        connection or none if connection failed
    """

    global POSTGRESS_CONNECTION

    if POSTGRESS_CONNECTION is None:
        POSTGRESS_CONNECTION = get_postgress_connection()

    return POSTGRESS_CONNECTION


def get_postgress_connection() -> psycopg2.extensions.connection | None:
    """ 
    Connect to the PostgreSQL db

    Returns:
        connection or none if connection unsuccessful
    """
    conn_string = (
        f"host={str(db_host)}"
        f"port={str(db_port)}"
        f"dbname={str(database)}"
        f"user={str(user)}"
        f"password={str(password())}"
    )

    try:
        connection = psycopg2.connect(conn_string)
        connection.autocommit = True
        logger.info("DB connection successful")
        return connection
    except Exception as e:
        logger.exception("DB connection failed")
        return None
    

def get_user_classes(user_id: int,
                     table_name: str = "kyc_bot_email_processing") -> list[str] | None:
    """ 
    Read classes defined by the given user from the postgresql db

    Args:
        user_id: the unique user identifier
        table_name: the table name to get classes from

    Returns:
        list of user classses
    """

    if table_name != "kyc_bot_email_processing":
        raise NotImplementedError
    
    final_result = []
    connection = postgress_connection()
    if connection is None:
        return "Failed to connect to the DB"
    
    with connection.cursor() as curs:
        curs.execute(
            """SELECT email_classes FROM kyc_bot_email_processing WHERE user_id=%(user_id)s;""",
            {"user_id": user_id},
        )
        
        result = curs.fetchall()

        if result:
            final_result = [item[0] for item in result]
            assert (len(final_result)==1), f"Found more than one record for user {user_id} in db."
            final_result = final_result[0]

    return final_result


def add_user_classes(
        user_id: int,
        classes: list[str],
        table_name: str = "kyc_bot_email_processing"
) -> dict[str, bool]:
    """ 
    Adds a new user defined class for given user to the postgresql db

    Args:
        user_id: the user unique identifier
        classes: classes to be added
        table_name: table name for which classes should be added
    
    Returns:
        bool values presenting whether adding the class was successful or not
    """
    if table_name != "kyc_bot_email_processing":
        raise NotImplementedError

    results = {}
    # check whether the user exists
    connection = postgress_connection()
    if connection is None:
        return "Failed to connect to the DB"

    with connection.cursor() as curs:
        curs.execute(
            """SELECT EXISTS(SELECT 1 FROM kyc_bot_email_processing WHERE user_id=%(user_id)s);""",
            {"user_id": user_id},
        ) 
        exists = curs.fetchone()[0]

    if not exists:
        # create new user and add classes
        with connection.cursor() as curs:
            curs.execute(
                """INSERT into kyc_bot_email_processing (user_id, email_classes) VALUES (%(user_id)s, %(classes)s);""",
                {
                    "user_id": user_id, 
                    "classes": "{" + (",").join(classes) + "}",
                },
            )
            connection.commit()
            for cls in classes:
                results[cls] = True 
    
    return results

    user_classes = get_user_classes(user_id=user_id, 
                                    table_name=table_name)
    # remove duplicated classes
    for cls in user_classes:
        if cls in classes:
            results[cls] = False
            classes.pop(classes.index(cls))

    # append classes to user data
    with connection.cursor() as curs:
        for cls in classes:
            curs.execute(
                """UPDATE kyc_bot_email_processing SET email_classes = array_append(email_classes,%(email_cls)s) WHERE user_id=%(user_id)s;""",
                {
                    "email_cls": cls,
                    "user_id": user_id,
                }
            )
            results[cls] = True

        connection.commit()

    return results


def remove_classes(
        user_id: int,
        classes: list[str],
        table_name: str = "kyc_bot_email_processing",
) -> dict[str, bool]:
    """ 
    Remove classes from the user defined set of classes at postgresql db

    Args:
        user_id: the user unique identifier
        classes: classes to be removed
        table_name: table name for which classes should be removed
    Returns:
        bool values presenting whether removing the class was successful or not
    """

    if table_name != "kyc_bot_email_processing":
        raise NotImplementedError
    
    results = {}
    connection = postgress_connection()
    if connection is None:
        return "Failed to connect to the DB"

    with connection.cursor() as curs:
        for cls in classes:
            curs.execute(
                """UPDATE kyc_bot_email_processing SET email_classes = array_remove(email_classes,%(email_cls)s) WHERE user_id=%(user_id)s;""",
                {
                    "email_cls": cls,
                    "user_id": user_id,
                }
            )
            results[cls] = True

        connection.commit()

    return results