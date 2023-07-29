from typing import Any, Optional
from dataclasses import dataclass, field
import os

from supabase import Client, create_client
from gotrue.types import AuthResponse, User, Session
from dotenv import load_dotenv

from .store import TensorStore


load_dotenv()


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

    def refresh(self) -> AuthResponse:
        # refresh
        response = self.client.auth.refresh_session(self._session.refresh_token)

        # renew tokens
        self._session = response.session

        # return response
        return response

    def __call__(self) -> TensorStore:
        # login if not logged in
        if not hasattr(self, '_session') or self._session is None:
            self.login_by_mail()

        # init a store
        return TensorStore(self)


def login(email: str, password: str, backend_url: Optional[str] = None, backend_key: Optional[str] = None) -> TensorStore:
    # load backend url and key from env
    if backend_url is None:
        backend_url = os.environ.get('SUPABASE_URL', 'http://localhost:8000')
    
    # TODO: replace default key with the hydrocode.cloud key
    if backend_key is None:
        try:
            backend_key = os.environ['SUPABASE_KEY']
        except KeyError:
            raise RuntimeError('No backend key provided and no SUPABASE_KEY environment variable found.')
    
    # init a session
    session = BackendSession(email, password, backend_url, backend_key)

    # login
    session.login_by_mail()

    # return session
    return session()
