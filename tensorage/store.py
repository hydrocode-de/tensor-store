from typing import TYPE_CHECKING, Tuple, Union, List
from dataclasses import dataclass, field

import numpy as np

if TYPE_CHECKING:
    from tensorage.session import BackendSession
from .types import Dataset


@dataclass
class TensorStore(object):
    _session: 'BackendSession' = field(repr=False)

    def __getitem__(self, key: Union[str, Tuple[Union[str, slice, int]]]):
        # first get key
        if isinstance(key, str):
            name = key
        elif isinstance(key[0], str):
            name = key[0]
        else:
            raise KeyError('You need to pass the key as first argument.')
        
        # load the dataset
        with self._session as context:
            dataset = context.get_dataset(name)

        # now we need to figure out, what kind of slice we need to pass
        if isinstance(key, str):
            index = [1, dataset.shape[0] + 1]
            slices = [[1, dataset.shape[i] + 1] for i in range(1, dataset.ndim)]
        
        # handle all the tuple cases
        else:
            # index
            if isinstance(key[1], int):
                index = [key[1] + 1, key[1] + 2]
            elif isinstance(key[1], slice):
                index = [key[1].start + 1, key[1].stop + 2]
            else:
                raise KeyError('Batch index needs to be passed as int or slice.')
            
            # slices
            if len(key) == 2:
                slices = [[1, dataset.shape[i] + 1] for i in range(2, dataset.ndim)]
            else:  # more than 2
                slices = []
                for i, arg in enumerate(key[2:]):
                    if isinstance(arg, int):
                        slices.append([arg + 1, arg + 1])
                    elif isinstance(arg, slice):
                        slices.append([arg.start + 1 if arg.start is not None else 1, arg.stop + 1 if arg.stop is not None else dataset.shape[i + 1] + 1])
                    else:
                        raise KeyError('Slice needs to be passed as int or slice.')
                
                # check if we have all slices
                if len(slices) + 1 != dataset.ndim:
                    for i in range(len(slices) + 1, dataset.ndim):
                        slices.append([1, dataset.shape[i] + 1])
        
        # now, name, index and slices are set
        with self._session as context:
            # load the tensor
            arr = context.get_tensor(name, index[0], index[1], [s[0] for s in slices], [s[1] for s in slices])
        
        # TODO now we can transform to other libaries
        return arr

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

    def __delitem__(self, key: str):
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
        # setup auth token
        self.__setup_auth()
        
        # run the insert
        try:
            response = self.backend.client.table('tensors_float4').insert([{'data_id': data_id, 'index': i + 1, 'user_id': self.user_id, 'tensor': chunk.tolist()} for i, chunk in enumerate(data)]).execute()
        except Exception as e:
            print(response.data)
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
        return Dataset(id=data['id'], key=data['key'], shape=data['shape'], ndim=data['ndim'])

    def get_tensor(self, name: str, index_low: int, index_up: int, slice_low: List[int], slice_up: List[int]) -> np.ndarray:
        # setup auth token
        self.__setup_auth()

        # get the requested chunk
        response = self.backend.client.rpc('tensor_float4_slice', {'name': name, 'index_low': index_low, 'index_up': index_up, 'slice_low': slice_low, 'slice_up': slice_up}).execute()

        # restore old token
        self.__restore_auth()

        # grab the data
        data = response.data[0]['tensor']

        # return as np.ndarray
        return np.asarray(data)

    def remove_dataset(self, key: str):
        raise NotImplementedError

    def list_tensor_keys(self):
        raise NotImplementedError

    def __del__(self):
        self.backend.logout()
