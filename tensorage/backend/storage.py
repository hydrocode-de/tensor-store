from typing import List, Tuple
import os

import numpy as np
from storage3.utils import StorageException

from tensorage.types import Dataset

from .base import BaseContext

class StorageContext(BaseContext):
    def __setup_auth(self):
        # store the current JWT token
        self._anon_key = self.backend.client.supabase_key

        # add the authenticated JWT to the headers
        self.backend.client.storage._client.headers['Authorization'] = f"Bearer {self.backend._session.access_token}"

    def __post_init__(self):
        if not self.has_bucket():
            self._create_user_bucket()

    def __restore_auth(self):
        # restore the original JWT
        self.backend.client.storage._client.headers['Authorization'] = f"Bearer {self._anon_key}"

    def _create_user_bucket(self) -> bool:
        # setup auth token
        self.__setup_auth()

        # create a lookup for all accessible buckets
        lookup = {buck.name: buck.id for buck in self.backend.client.storage.list_buckets()}
        
        # create the bucket
        res = self.backend.client.storage.create_bucket(id=self.user_id, name=self.backend._user.email)
        
        # restore the original auth token
        self.__restore_auth()

        return 'error' not in res

    def has_bucket(self) -> bool:
        # setup auth token
        self.__setup_auth()

        # try to find the bucket
        try:
            bucket = self.backend.client.storage.get_bucket(self.user_id)
        except StorageException as e:
            return False
        return True
    
    def get_dataset(self, key: str) -> Dataset:
        return super().get_dataset(key)
    
    def get_tensor(self, key: str, index_low: int, index_up: int, slice_low: List[int], slice_up: List[int]) -> np.ndarray:
        return super().get_tensor(key, index_low, index_up, slice_low, slice_up)
    
    def insert_dataset(self, key: str, shape: Tuple[int], dim: int, type: str, is_shared: bool) -> Dataset:
        return super().insert_dataset(key, shape, dim, type, is_shared)
    
    def insert_tensor(self, data_id: int, data: List[np.ndarray], offset: int = 0) -> bool:
        return super().insert_tensor(data_id, data, offset)
    
    def append_tensor(self, data_id: int, data: List[np.ndarray]) -> bool:
        return super().append_tensor(data_id, data)
    
    def remove_dataset(self, key: str) -> bool:
        return super().remove_dataset(key)
    
    def list_dataset_keys(self) -> List[str]:
        return super().list_dataset_keys()
