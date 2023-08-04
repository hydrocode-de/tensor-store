from typing import Tuple, List

from postgrest.exceptions import APIError
import numpy as np

from tensorage.types import Dataset
from .base import BaseContext



class DatabaseContext(BaseContext):
    def __setup_auth(self):
        # store the current JWT token
        self._anon_key = self.backend.client.supabase_key

        # set the JWT of the authenticated user as the new token
        self.backend.client.postgrest.auth(self.backend._session.access_token)
    
    def __restore_auth(self):
        # restore the original JWT
        self.backend.client.postgrest.auth(self._anon_key)

    def check_schema_installed(self) -> bool:
        # setup auth token
        self.__setup_auth()

        # check if the datasets and tensor_float4 tables exist
        missing_table = False

        for table in ('datasets', 'tensors_float4'):
            try:
                self.backend.client.table(table).select('*', count='exact').limit(1).execute()
            except APIError as e:
                if e.code == '42P01':
                    missing_table = True
                else:
                    raise e
        
        # check if any of the needed tables was not found
        return not missing_table

    def insert_dataset(self, key: str, shape: Tuple[int], dim: int) -> Dataset:
        # run the insert
        self.__setup_auth()
        response = self.backend.client.table('datasets').insert({'key': key, 'shape': shape, 'ndim': dim, 'user_id': self.user_id}).execute()
        self.__restore_auth()

        # return an instance of Dataset
        data = response.data[0]
        return Dataset(id=data['id'], key=data['key'], shape=data['shape'], ndim=data['ndim'])
    
    def insert_tensor(self, data_id: int, data: List[np.ndarray], offset: int = 0) -> bool:
        # setup auth token
        self.__setup_auth()

        # run the insert
        try:
            self.backend.client.table('tensors_float4').insert([{'data_id': data_id, 'index': int(i + 1 + offset), 'user_id': self.user_id, 'tensor': chunk.tolist()} for i, chunk in enumerate(data)]).execute()
        except APIError as e:
            # TODO check if we expired here and refresh the token
            raise e
        
        # restore old token
        self.__restore_auth()

        # return 
        return True

    def get_dataset(self, key: str) -> Dataset:
        # setup auth token
        self.__setup_auth()

        # get the dataset
        response = self.backend.client.table('datasets').select('*').eq('key', key).execute()

        # restore old token
        self.__restore_auth()

        # grab the data
        data = response.data[0]

        # return as Dataset
        # TODO -> here we hardcode the type to float32 as nothing else is implemented so far
        return Dataset(id=data['id'], key=data['key'], shape=data['shape'], ndim=data['ndim'], is_shared=data['is_shared'], type='float32')

    def get_tensor(self, key: str, index_low: int, index_up: int, slice_low: List[int], slice_up: List[int]) -> np.ndarray:
        # setup auth token
        self.__setup_auth()

        # get the requested chunk
        response = self.backend.client.rpc('tensor_float4_slice', {'name': key, 'index_low': index_low, 'index_up': index_up, 'slice_low': slice_low, 'slice_up': slice_up}).execute()

        # restore old token
        self.__restore_auth()

        # grab the data
        data = response.data[0]['tensor']

        # return as np.ndarray
        return np.asarray(data)

    def remove_dataset(self, key: str) -> bool:
        # setup auth token
        self.__setup_auth()

        # remove the dataset
        self.backend.client.table('datasets').delete().eq('key', key).execute()

        # restore old token
        self.__restore_auth()
        
        # return
        return True

    def list_dataset_keys(self) -> List[str]:
        # setup auth token
        self.__setup_auth()

        # get the keys
        response = self.backend.client.table('datasets').select('key').execute()

        # restore old token
        self.__restore_auth()

        return [row['key'] for row in response.data]

    def append_tensor(self, data_id: int, data: List[np.ndarray]) -> bool:
        return super().append_tensor(data_id, data)
