from typing import Optional, Tuple
import os
import json

from gotrue.types import AuthResponse

from .store import TensorStore
from .session import BackendSession


# supabase connection file
SUPA_FILE = os.path.join(os.path.dirname(__file__), '.supabase.env')


def __get_auth_info(backend_url: Optional[str], backend_key: Optional[str] = None) -> Tuple[str, str]:
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
    # get the environment variables
    backend_url, backend_key = __get_auth_info(backend_url=backend_url, backend_key=backend_key)
    
    # get a session
    session = BackendSession(email, password, backend_url, backend_key)

    # bind the session to the Store
    store = TensorStore(session)

    # return the store
    return store


def signup(email: str, password: str, backend_url: Optional[str] = None, backend_key: Optional[str] = None) -> AuthResponse:
    # get the environment variables
    backend_url, backend_key = __get_auth_info(backend_url=backend_url, backend_key=backend_key)
        
    # get a session
    session = BackendSession(None, None, backend_url, backend_key)

    # register
    response = session.register_by_mail(email, password)
    return response
