"""
This module provides authentication and authorization functionality for Supabase.

It provides a `BackendSession` class for managing backend sessions, as well as utility functions for working with authentication tokens and Supabase connection information.

"""

from typing import Optional, Tuple, Union, overload
from typing_extensions import Literal
import os
import json

from gotrue.types import AuthResponse

from .store import TensorStore
from .session import BackendSession


# supabase connection file
SUPA_FILE = os.path.join(os.path.expanduser('~'), '.tensorage.conf')


def __get_auth_info(backend_url: Optional[str], backend_key: Optional[str] = None, email: Optional[str] = None, password: Optional[str] = None) -> Tuple[str, str, str, str]:
    """
    Get the Supabase connection information.

    This function returns the Supabase connection information as a tuple of the backend URL and backend key. If the connection information is not provided as arguments, it is read from the `.supabase.env` file or from environment variables.

    :param backend_url: The URL of the Supabase backend.
    :param backend_key: The API key for the Supabase backend.
    :param email: The email address of the user to log in.
    :param password: The password of the user to log in.
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
    
    if email is None:
        email = persisted.get('USER_EMAIL', os.environ.get('USER_EMAIL'))
    
    if password is None:
        password = persisted.get('USER_PASSWORD', os.environ.get('USER_PASSWORD'))

    # the supabase key may be None, we raise an exception in that case
    if backend_key is None:
        raise RuntimeError('SUPABASE_KEY environment variable not set and no KEY has been persisted.')
    
    # if there was no error, return
    return backend_url, backend_key, email, password


@overload
def link_to(backend_url: str, backend_key: str) -> Literal[True]:
    ...
@overload
def link_to(backend_url: str, backend_key: str, password: str, email: str) -> TensorStore:
    ...
def link_to(backend_url: str, backend_key: str, password: Optional[str] = None, email: Optional[str] = None) -> Union[Literal[True], TensorStore]:
    """
    Link to a Supabase backend using the provided backend URL and key.

    If a password and email are provided, logs in to the backend using the provided credentials.

    Args:
        backend_url (str): The URL of the Supabase backend to link to.
        backend_key (str): The key of the Supabase backend to link to.
        password (Optional[str]): The password to use for logging in to the backend. Defaults to None.
        email (Optional[str]): The email to use for logging in to the backend. Defaults to None.

    Returns:
        Union[True, TensorStore]: If no credentials are provided, returns True. If credentials are provided, returns a TensorStore object representing the linked backend.

    Raises:
        ValueError: If the backend URL or key are not provided.
        AuthenticationError: If the provided credentials are invalid.
    """
    # create the dict of connection information
    if password is not None and email is not None:
        persisted = dict(SUPABASE_URL=backend_url, SUPABASE_KEY=backend_key, USER_EMAIL=email, USER_PASSWORD=password)
    else:
        persisted = dict(SUPABASE_URL=backend_url, SUPABASE_KEY=backend_key)

    # create or overwrite the SUPA_FILE
    with open(SUPA_FILE, 'w') as f:
        json.dump(persisted, f)
    
    # if the user provided a password and email, we log in
    if password is not None and email is not None:
        return login(email, password, backend_url, backend_key)
    else:
        return True


def login(email: Optional[str] = None, password: Optional[str] = None, backend_url: Optional[str] = None, backend_key: Optional[str] = None) -> TensorStore:
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
    backend_url, backend_key, email, password = __get_auth_info(backend_url=backend_url, backend_key=backend_key, email=email, password=password)
    
    # check that email and password are supplied
    if email is None or password is None:
        raise RuntimeError(f"Email and password are not saved in {SUPA_FILE} and must therfore be supplied for login.")
    
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
