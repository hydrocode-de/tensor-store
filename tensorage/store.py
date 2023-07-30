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

    def __setitem__(self, key: str, value: Union[List[list], np.ndarray]):
        # first make a numpy array from it
        if isinstance(value, list):
            value = np.asarray(value)

        # make at least 2D 
        if value.ndim == 1:
            value = value.reshape(1, -1)        
        
        # get the shape
        shape = value.shape

        # get the dim
        dim = value.ndim

        # connect
        with self._session as context:
            # insert the dataset
            dataset = context.insert_dataset(key, shape, dim)

            # insert the tensor
            context.insert_tensor(dataset.id, [chunk for chunk in value])

    def __delitem__(self, key):
        raise NotImplementedError

    def keys(self):
        raise NotImplementedError


@dataclass
class StoreContext(object):
    # This is the OPENED session, as we are in the StoreContext
    backend: 'BackendSession' = field(repr=False)
    _anon_key: str = field(init=False, repr=False)

    def __setup_auth(self):
        # store the current JWT token
        self._anon_key = self.backend.client.supabase_key

        # set the JWT of the authenticated user as the new token
        self.backend.client.postgrest.auth(self.backend._session.access_token)
    
    def __restore_auth(self):
        # restore the original JWT
        self.backend.client.postgrest.auth(self._anon_key)

    @property
    def user_id(self) -> str:
        return self.backend._user.id

    def insert_dataset(self, key: str, shape: Tuple[int], dim: int) -> Dataset:
        # run the insert
        self.__setup_auth()
        response = self.backend.client.table('datasets').insert({'key': key, 'shape': shape, 'ndim': dim, 'user_id': self.user_id}).execute()
        self.__restore_auth()

        # return an instance of Dataset
        data = response.data[0]
        return Dataset(id=data['id'], key=data['key'], shape=data['shape'], ndim=data['ndim'])
    
    def insert_tensor(self, data_id: int, data: List[np.ndarray]) -> bool:
        # run the insert
        self.__setup_auth()
        try:
            response = self.backend.client.table('tensors_float4').insert([{'data_id': data_id, 'index': i + 1, 'user_id': self.user_id, 'tensor': chunk.tolist()} for i, chunk in enumerate(data)]).execute()
        except Exception as e:
            print(response.data)
            raise e
        
        self.__restore_auth()

        # return 
        return True

    def get_tensor():
        raise NotImplementedError

    def remove_tensor():
        raise NotImplementedError

    def list_tensor_keys():
        raise NotImplementedError

    def __del__(self):
        self.backend.logout()
