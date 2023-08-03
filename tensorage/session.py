from typing import Any, TypeVar, Generic
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
    _session: 'BackendSession'
    Context: type[C]

    def __enter__(self) -> C:
        # login the session if it is not logged in
        if not hasattr(self._session, '_session') or self._session._session is None:
            self._session.login_by_mail()
        
        # instatiate the store with an authenticated Session
        context = self.Context(self._session)

        return context

    def __exit__(self, *args):
        # logout the session
        self._session.logout()


@dataclass
class BackendSession(object):
    email: str
    password: str
    backend_url: str 
    backend_key: str = field(repr=False)
    _client: Client = field(init=False, repr=False)
    _user: User = field(init=False, repr=False)
    _session: Session = field(init=False, repr=False)


    @property
    def client(self) -> Client:
        if not hasattr(self, '_client') or self._client is None:
            self._client = create_client(self.backend_url, self.backend_key)
        return self._client

    def login_by_mail(self) -> AuthResponse:
        # login
        response = self.client.auth.sign_in_with_password({'email': self.email, 'password': self.password})

        # store user and session info
        self._user = response.user
        self._session = response.session
        # return response
        return response
    
    def register_by_mail(self, email: str, password: str) -> AuthResponse:
        # register
        response = self.client.auth.sign_up({'email': email, 'password': password})
        
        # store user and session info
        self._user = response.user
        self._session = response.session

        # return response
        return response

    def refresh(self) -> AuthResponse:
        # refresh
        response = self.client.auth.refresh_session(self._session.refresh_token)

        # renew tokens
        self._session = response.session

        # return response
        return response
    
    def logout(self):
        if hasattr(self, '_client') and self._client is not None:
            self.client.auth.sign_out()

    def database(self) -> ContextWrapper[DatabaseContext]:
        return ContextWrapper(self, DatabaseContext)
    
    def storage(self) -> ContextWrapper[StorageContext]:
        return ContextWrapper(self, StorageContext)

    def __del__(self):
        self.logout()

    def __call__(self) -> TensorStore:
        # init a store
        return TensorStore(self)
