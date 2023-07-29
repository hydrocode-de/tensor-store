from typing import TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from tensor_store.session import BackendSession


@dataclass
class TensorStore(object):
    _session: 'BackendSession' = field(repr=False)

    def __getitem__(self, key):
        raise NotImplementedError

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __delitem__(self, key):
        raise NotImplementedError

    def keys(self):
        raise NotImplementedError


@dataclass
class StoreContext(object):
    _session: 'BackendSession' = field(repr=False)

    def add_new_tensor():
        raise NotImplementedError

    def get_tensor():
        raise NotImplementedError

    def remove_tensor():
        raise NotImplementedError

    def list_tensor_keys():
        raise NotImplementedError

    def __del__(self):
        self._session.logout()
