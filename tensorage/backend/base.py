from typing import TYPE_CHECKING, List, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

if TYPE_CHECKING:
    from tensorage.session import BackendSession
    from tensorage.types import Dataset    


@dataclass
class BaseContext(ABC):
    backend: 'BackendSession' = field(repr=False)
    _anon_key: str = field(init=False, repr=False)

    @property
    def user_id(self) -> str:
        return self.backend._user.id

    @abstractmethod
    def get_dataset(self, key: str) -> 'Dataset':
        raise NotImplementedError

    @abstractmethod
    def get_tensor(self, key: str, index_low: int, index_up: int, slice_low: List[int], slice_up: List[int]) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def insert_dataset(self, key: str, shape: Tuple[int], dim: int, type: str, is_shared: bool) -> 'Dataset':
        raise NotImplementedError
    
    @abstractmethod
    def insert_tensor(self, data_id: int, data: List[np.ndarray], offset: int = 0) -> bool:
        raise NotImplementedError

    @abstractmethod
    def append_tensor(self, data_id: int, data: List[np.ndarray]) -> bool:
        raise NotImplementedError
    
    @abstractmethod
    def remove_dataset(self, key: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def list_dataset_keys(self) -> List[str]:
        raise NotImplementedError
    
    def __del__(self):
        self.backend.logout()