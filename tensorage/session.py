"""
This module provides the `BackendSession` class, which is used to manage the 
backend session for the application.

The `BackendSession` class provides methods for logging in and out of the backend, 
as well as for accessing the various backend contexts (database, storage, etc.). 
It also provides a context manager (`ContextWrapper`) for managing the 
lifetime of backend contexts.

Example:
    .. code-block:: python

        session = BackendSession()
        with session.database() as db:
            db.insert_dataset(key='test', shape=[1, 2, 3], dim=3)

"""

from typing import Any, TypeVar, Generic, Type
from dataclasses import dataclass, field

from supabase import Client, create_client
from gotrue.types import AuthResponse, User, Session
from dotenv import load_dotenv

from .store import TensorStore
from .backend.base import BaseContext
from .backend.database import DatabaseContext
from .backend.storage import StorageContext

load_dotenv()


C = TypeVar('C', bound=BaseContext)

@dataclass
class ContextWrapper(Generic[C]):
    """
    A context manager for managing the lifetime of backend contexts.

    The `ContextWrapper` class is a generic context manager that takes a 
    backend context class as a type parameter. It provides a `__enter__` method 
    that logs in the backend session if it is not already logged in, 
    and instantiates the backend context with an authenticated session. 
    It also provides a `__exit__` method that logs out the backend session.

    Example:
        .. code-block:: python

            session = BackendSession()
            with ContextWrapper(session, DatabaseContext) as db:
                db.insert_dataset(key='test', shape=[1, 2, 3], dim=3)

    """
    _session: 'BackendSession'
    Context: Type[C]

    def __enter__(self) -> C:
        """
        Enter the context manager and return the backend context instance.

        If the backend session is not already logged in, this method logs 
        in the session using the `login_by_mail` method of the `BackendSession` 
        instance. It then instantiates the backend context with an authenticated 
        session, and returns the context instance.

        Example:
            .. code-block:: python

                session = BackendSession()
                with ContextWrapper(session, DatabaseContext) as db:
                    db.insert_dataset(key='test', shape=[1, 2, 3], dim=3)

        :return: The backend context instance.
        """
        # login the session if it is not logged in
        if not hasattr(self._session, '_session') or self._session._session is None:
            self._session.login_by_mail()
        
        # instatiate the store with an authenticated Session
        context = self.Context(self._session)

        return context

    def __exit__(self, *args):
        """
        Exit the context manager and log out the backend session.

        This method logs out the backend session using the `logout` method of the `BackendSession` instance.

        Example:
            .. code-block:: python

                session = BackendSession()
                with ContextWrapper(session, DatabaseContext) as db:
                    db.insert_dataset(key='test', shape=[1, 2, 3], dim=3)

        :param args: The exception type, value, and traceback (if any).
        """        # logout the session
        self._session.logout()


@dataclass
class BackendSession(object):
    """
    A class for managing the backend session for the application.

    The `BackendSession` class provides methods for logging in and out of the backend, 
    as well as for accessing the various backend contexts (database, storage, etc.). 
    It also provides a context manager (`ContextWrapper`) for managing the lifetime of backend contexts.

    Example:
        .. code-block:: python

            session = BackendSession()
            session.login_by_mail('user@example.com', 'password')
            with session.database() as db:
                db.insert_dataset(key='test', shape=[1, 2, 3], dim=3)

    :ivar client: The backend client instance.
    :ivar token: The authentication token for the backend session.
    """
    email: str
    password: str
    backend_url: str 
    backend_key: str = field(repr=False)
    _client: Client = field(init=False, repr=False)
    _user: User = field(init=False, repr=False)
    _session: Session = field(init=False, repr=False)


    @property
    def client(self) -> Client:
        """
        Get the backend client instance.

        If the client instance has not been created yet, this method creates it using the `Client` class from the `postgrest` module. It then sets the `Authorization` header of the client instance to the authentication token for the backend session.

        Example:
            .. code-block:: python

                session = BackendSession()
                session.login_by_mail('user@example.com', 'password')
                client = session.client()
                response = client.table('datasets').select('*').execute()

        :return: The backend client instance.
        """
        if not hasattr(self, '_client') or self._client is None:
            self._client = create_client(self.backend_url, self.backend_key)
        return self._client

    def login_by_mail(self) -> AuthResponse:
        """
        Log in to the backend using email and password.

        This method logs in to the backend using the `auth.sign_in` method of the `supabase.Client` instance. 
        Email and password are used from the `BackendSession` object, and returns an `AuthResponse` object 
        containing the authentication token and user information.

        :return: An `AuthResponse` object containing the authentication token and user information.
        """
        # login
        response = self.client.auth.sign_in_with_password({'email': self.email, 'password': self.password})

        # store user and session info
        self._user = response.user
        self._session = response.session
        # return response
        return response
    
    def register_by_mail(self, email: str, password: str) -> AuthResponse:
        """
        Register a new user account using email and password.

        This method registers a new user account using the `auth.sign_up` method of the `supabase.Client` instance.
        It takes an email and password as arguments, and returns an `AuthResponse` object containing the 
        authentication token and user information.

        Example:
            .. code-block:: python

                session = BackendSession()
                response = session.register_by_mail('user@example.com', 'password')
                print(response)

        :param email: The email address of the user.
        :param password: The password of the user.
        :return: An `AuthResponse` object containing the authentication token and user information.
        """        # register
        response = self.client.auth.sign_up({'email': email, 'password': password})
        
        # store user and session info
        self._user = response.user
        self._session = response.session

        # return response
        return response

    def refresh(self) -> AuthResponse:
        """
        Refresh the authentication token for the backend session.

        This method refreshes the authentication token for the backend session using the `auth.refresh_access_token` method of the `supabase.Client` instance. It returns an `AuthResponse` object containing the new authentication token and user information.

        Example:
            .. code-block:: python

                session = BackendSession()
                session.login_by_mail('user@example.com', 'password')
                response = session.refresh()
                print(response)

        :return: An `AuthResponse` object containing the new authentication token and user information.
        """
        # refresh
        response = self.client.auth.refresh_session(self._session.refresh_token)

        # renew tokens
        self._session = response.session

        # return response
        return response
    
    def logout(self):
        """
        Log out of the backend session.

        This method logs out of the backend session using the `auth.sign_out` method of the `supabase.Client` instance.

        Example:
            .. code-block:: python

                session = BackendSession()
                session.login_by_mail('user@example.com', 'password')
                session.logout()

        """
        if hasattr(self, '_client') and self._client is not None:
            self.client.auth.sign_out()

    def database(self) -> ContextWrapper[DatabaseContext]:
        """
        Get a context manager for the database context.

        This method returns a context manager (`ContextWrapper`) for the database context (`DatabaseContext`). 
        The context manager handles the lifetime of the database context, including logging in and out of 
        the backend session.

        Example:
            .. code-block:: python

                session = BackendSession()
                with session.database() as db:
                    db.insert_dataset(key='test', shape=[1, 2, 3], dim=3)

        :return: A context manager for the database context.
        """
        return ContextWrapper(self, DatabaseContext)
    
    def storage(self) -> ContextWrapper[StorageContext]:
        """
        Get a context manager for the storage context.

        This method returns a context manager (`ContextWrapper`) for the storage context (`StorageContext`). 
        The context manager handles the lifetime of the storage context, including logging in and out 
        of the backend session.

        Example:
            .. code-block:: python

                session = BackendSession()
                with session.storage() as storage:
                    storage.upload_file('test.txt', 'Hello, world!')

        :return: A context manager for the storage context.
        """
        return ContextWrapper(self, StorageContext)

    def __del__(self):
        """
        Clean up the backend session when the object is deleted.

        This method logs out of the backend session and cleans up any resources associated 
        with the session when the `BackendSession` object is deleted.

        Example:
            .. code-block:: python

                session = BackendSession()
                session.login_by_mail('user@example.com', 'password')
                del session

        """
        self.logout()

    def __call__(self) -> TensorStore:
        """
        Get the tensor store instance for the backend session.

        If the tensor store instance has not been created yet, this method creates it using the 
        `TensorStore` class from the `tensorstore` module. It then sets the `Authorization` 
        header of the tensor store instance to the authentication token for the backend session.

        Example:
            .. code-block:: python

                session = BackendSession()
                session.login_by_mail('user@example.com', 'password')
                store = session()
                array = np.array([[1, 2], [3, 4]])
                store['test'] = array

        :return: The tensor store instance for the backend session.
        """
        # init a store
        return TensorStore(self)
