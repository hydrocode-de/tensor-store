"""
This module provides authentication and authorization functionality for Supabase.

It provides a `BackendSession` class for managing backend sessions, as well as utility functions for working with authentication tokens and Supabase connection information.

"""

from typing import Optional, Tuple
import os
import json

from gotrue.types import AuthResponse

from .store import TensorStore
from .session import BackendSession


# supabase connection file
SUPA_FILE = os.path.join(os.path.dirname(__file__), '.supabase.env')


def __get_auth_info(backend_url: Optional[str], backend_key: Optional[str] = None) -> Tuple[str, str]:
    """
    Get the Supabase connection information.

    This function returns the Supabase connection information as a tuple of the backend URL and backend key. If the connection information is not provided as arguments, it is read from the `.supabase.env` file or from environment variables.

    :param backend_url: The URL of the Supabase backend.
    :param backend_key: The API key for the Supabase backend.
    :return: A tuple of the backend URL and backend key.
    """
    # check if we saved persisted connection information
    if os.path.exists(SUPA_FILE):
        with open(SUPA_FILE, 'r') as f:
            persisted = json.load(f)
    else:
        persisted = dict()
    
    # if the user supplied url and key, we do not overwrite them
    if backend_url is None:
        backend_url = persisted.get('SUPABASE_URL', os.environ.get('SUPABASE_URL', 'http://localhost:8000'))
    
    if backend_key is None:
        backend_key = persisted.get('SUPABASE_KEY', os.environ.get('SUPABASE_KEY'))

    # the supabase key may be None, we raise an exception in that case
    if backend_key is None:
        raise RuntimeError('SUPABASE_KEY environment variable not set and no KEY has been persisted.')
    
    # if there was no error, return
    return backend_url, backend_key


def login(email: str, password: str, backend_url: Optional[str] = None, backend_key: Optional[str] = None) -> TensorStore:
    """
    Log in to the Supabase backend using email and password authentication.

    This function creates a `BackendSession` object using the provided backend URL and key, or the default values if none are provided. It then logs in to the backend session using the provided email and password. If the login is successful, it returns the tensor store instance for the backend session.

    :param email: The email address of the user to log in.
    :param password: The password of the user to log in.
    :param backend_url: The URL of the Supabase backend. Defaults to `None`.
    :param backend_key: The API key for the Supabase backend. Defaults to `None`.
    :return: The tensor store instance for the backend session.
    :raises RuntimeError: If the login fails.
    """
    # get the environment variables
    backend_url, backend_key = __get_auth_info(backend_url=backend_url, backend_key=backend_key)
    
    # get a session
    session = BackendSession(email, password, backend_url, backend_key)

    # bind the session to the Store
    store = TensorStore(session)

    # return the store
    return store


def signup(email: str, password: str, backend_url: Optional[str] = None, backend_key: Optional[str] = None) -> AuthResponse:
    """
    Sign up a new user to the Supabase backend using email and password authentication.

    This function creates a `BackendSession` object using the provided backend URL and key, or the default values if none are provided. It then signs up a new user to the backend session using the provided email and password. If the signup is successful, it returns an `AuthResponse` object containing the user's access token and refresh token.

    :param email: The email address of the user to sign up.
    :param password: The password of the user to sign up.
    :param backend_url: The URL of the Supabase backend. Defaults to `None`.
    :param backend_key: The API key for the Supabase backend. Defaults to `None`.
    :return: An `AuthResponse` object containing the user's access token and refresh token.
    :raises RuntimeError: If the signup fails.
    """
    # get the environment variables
    backend_url, backend_key = __get_auth_info(backend_url=backend_url, backend_key=backend_key)
        
    # get a session
    session = BackendSession(None, None, backend_url, backend_key)

    # register
    response = session.register_by_mail(email, password)
    return response
