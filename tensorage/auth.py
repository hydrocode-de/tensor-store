from typing import Optional
import os

from gotrue.types import AuthResponse

from .store import TensorStore
from .session import BackendSession

def login(email: str, password: str, backend_url: Optional[str] = None, backend_key: Optional[str] = None) -> TensorStore:
    # get the environment variables
    if backend_url is None:
        backend_url = os.getenv('SUPABASE_URL', 'http://localhost:8000')
    
    if backend_key is None:
        try:
            backend_key = os.environ['SUPABASE_KEY']
        except KeyError:
            raise RuntimeError('SUPABASE_KEY environment variable not set')
    
    # get a session
    session = BackendSession(email, password, backend_url, backend_key)

    # bind the session to the Store
    store = TensorStore(session)

    # return the store
    return store


def register(email: str, password: str, backend_url: Optional[str] = None, backend_key: Optional[str] = None) -> AuthResponse:
    # get the environment variables
    if backend_url is None:
        backend_url = os.getenv('SUPABASE_URL', 'http://localhost:8000')

    if backend_key is None:
        try:
            backend_key = os.environ['SUPABASE_KEY']
        except KeyError:
            raise RuntimeError('SUPABASE_KEY environment variable not set')
        
    # get a session
    session = BackendSession(None, None, backend_url, backend_key)

    # register
    response = session.register_by_mail(email, password)
    return response
