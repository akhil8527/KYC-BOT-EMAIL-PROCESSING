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


def postgress_connection