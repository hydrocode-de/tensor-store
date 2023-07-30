from typing import TYPE_CHECKING, Tuple, Union, List
from dataclasses import dataclass, field

import numpy as np

if TYPE_CHECKING:
    from tensorage.session import BackendSession
from .types import Dataset


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

    def insert_dataset(self, key: str, shape: Tuple[int], dim: int) -> Dataset:
        raise NotImplementedError
    
    def insert_tensor(self, data_id: int, data: Union[List[list], np.ndarray]):
        raise NotImplementedError

    def get_tensor():
        raise NotImplementedError

    def remove_tensor():
        raise NotImplementedError

    def list_tensor_keys():
        raise NotImplementedError

    def __del__(self):
        self._session.logout()
