from typing import List, Tuple
import io
import json

import numpy as np
import xarray as xr
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
        # setup auth token
        self.__setup_auth()

        # download the metadata
        buf = io.BytesIO()
        try:
            content = self.backend.client.storage.from_(self.user_id).download(f"{key}/dataset.json")
            buf.write(content)
        except StorageException as e:
            if len(e.args) > 0 and e.args[0]['error'] == 'not_found':
                raise FileNotFoundError(f"Dataset with key '{key}' not found")
            else:
                raise e

        # rewind
        buf.seek(0)
        dataset = Dataset(**json.load(buf))

        # restore the original auth token
        self.__restore_auth()

        return dataset

    def get_tensor(self, key: str, index_low: int, index_up: int, slice_low: List[int], slice_up: List[int]) -> np.ndarray:
        return super().get_tensor(key, index_low, index_up, slice_low, slice_up)
    
    def insert_dataset(self, key: str, shape: Tuple[int], dim: int, type: str, is_shared: bool) -> Dataset:
        # setup auth token
        self.__setup_auth()

        # create the dataset metadata as a json
        metadata = json.dumps(dict(id=key, key=key, shape=shape, ndim=dim, type=type, is_shared=is_shared))

        # upload the metadata
        res = self.backend.client.storage.from_(self.user_id).upload(f"{key}/dataset.json", metadata.encode('utf-8'))

        # restore the original auth token
        self.__restore_auth()

        return res.json()
    
    def insert_tensor(self, data_id: int, data: List[np.ndarray], offset: int = 0) -> bool:
        # setup auth token
        self.__setup_auth()

        # upload each netcdf chunk
        for i, arr in enumerate(data):
            # build the netcdf
            netcdf = xr.Dataset({data_id: xr.DataArray(arr, dims=[f"dim_{i + 2}" for i in range(arr.ndim)])})
            
            # buffer and get the bytes
            buf = io.BytesIO()
            netcdf.to_netcdf(buf)
            buf.seek(0)
            b_netcdf = buf.getvalue()

            # upload the netcdf
            self.backend.client.storage.from_(self.user_id).upload(f"{data_id}/index_{int(i + 1 + offset)}.nc", b_netcdf)

        # restore the original auth token
        self.__restore_auth()

        return True

    def append_tensor(self, data_id: int, data: List[np.ndarray]) -> bool:
        return super().append_tensor(data_id, data)
    
    def remove_dataset(self, key: str) -> bool:
        # setup auth token
        self.__setup_auth()

        # remove the dataset
        self.backend.client.storage.from_(self.user_id).remove(f"{key}")

        # restore the original auth token
        self.__restore_auth()
        
        return True
    
    def list_dataset_keys(self) -> List[str]:
        return super().list_dataset_keys()
