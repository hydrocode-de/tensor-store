"""This module defines the DatabaseContext class which is responsible for interacting with the Supabase backend and its underlying Postgres database."""
from typing import Tuple, List

from postgrest.exceptions import APIError
import numpy as np

from tensorage.types import Dataset
from .base import BaseContext



class DatabaseContext(BaseContext):
    """
    A class representing a database context for interacting with the Supabase backend and its underlying Postgres database.
    """
    def __setup_auth(self):
        """
        Sets up the authentication for the Supabase client using the provided API key and URL.

        Raises:
            ValueError: If the API key or URL is not provided.
        """
        # store the current JWT token
        self._anon_key = self.backend.client.supabase_key

        # set the JWT of the authenticated user as the new token
        self.backend.client.postgrest.auth(self.backend._session.access_token)
    
    def __restore_auth(self):
        """
        Restores the authentication for the Supabase client using the stored anonymous key.

        Raises:
            ValueError: If the session token is not provided.
        """
        # restore the original JWT
        self.backend.client.postgrest.auth(self._anon_key)

    def check_schema_installed(self) -> bool:
        """
        Checks if the required schema is installed in the database.
        Right now, only the tables 'datasets' and 'tensors_float4' are checked for exitence.

        Returns:
            bool: True if the schema is installed, False otherwise.
        """
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
        """
        Inserts a new dataset into the database with the given key, shape, and dimension.

        Args:
            key (str): The unique identifier for the dataset.
            shape (Tuple[int]): The shape of the dataset.
            dim (int): The dimension of the dataset.

        Returns:
            Dataset: The newly created dataset object.
        """
        # run the insert
        self.__setup_auth()
        response = self.backend.client.table('datasets').insert({'key': key, 'shape': shape, 'ndim': dim, 'user_id': self.user_id}).execute()
        self.__restore_auth()

        # return an instance of Dataset
        data = response.data[0]
        return Dataset(id=data['id'], key=data['key'], shape=data['shape'], ndim=data['ndim'], is_shared=data['is_shared'], type=data['type'])
    
    def insert_tensor(self, data_id: int, data: List[np.ndarray], offset: int = 0) -> bool:
        """
        Inserts a tensor into the database with the given data ID, data, and offset.
        The offset is the position along the main (first) axis, at with the chunks given
        as data should be inserted.

        Args:
            data_id (int): The unique identifier for the tensor data.
            data (List[np.ndarray]): The tensor data to be inserted.
            offset (int): The offset to start inserting the tensor data.

        Returns:
            bool: True if the tensor data was successfully inserted, False otherwise.
        """
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
        """
        Retrieves the dataset with the given key from the database.

        Args:
            key (str): The unique identifier for the dataset.

        Returns:
            Dataset: The dataset object with the given key.

        Raises:
            ValueError: If the dataset with the given key does not exist in the database.
        """
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
        """
        Retrieves a tensor from the database with the given key, index range, and slice range.
        The index is the numeric index along the main axis, while the slice is marking the index ranges
        along the other axes. Please note, that all existing axes have to be covered, even if all data
        is requested.

        Args:
            key (str): The unique identifier for the tensor.
            index_low (int): The lower index bound for the tensor.
            index_up (int): The upper index bound for the tensor.
            slice_low (List[int]): The lower slice bound for the tensor.
            slice_up (List[int]): The upper slice bound for the tensor.

        Returns:
            np.ndarray: The tensor data with the given key, index range, and slice range.

        Raises:
            ValueError: If the tensor with the given key does not exist in the database.
        """        # setup auth token
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
        """
        Removes the dataset with the given key from the database.

        Args:
            key (str): The unique identifier for the dataset.

        Returns:
            bool: True if the dataset was successfully removed, False otherwise.

        Raises:
            ValueError: If the dataset with the given key does not exist in the database.
        """
        # setup auth token
        self.__setup_auth()

        # remove the dataset
        self.backend.client.table('datasets').delete().eq('key', key).execute()

        # restore old token
        self.__restore_auth()
        
        # return
        return True

    def list_dataset_keys(self) -> List[str]:
        """
        Retrieves a list of all dataset keys in the database.

        Returns:
            List[str]: A list of all dataset keys in the database.
        """
        # setup auth token
        self.__setup_auth()

        # get the keys
        response = self.backend.client.table('datasets').select('key').execute()

        # restore old token
        self.__restore_auth()

        return [row['key'] for row in response.data]

    def append_tensor(self, data_id: int, data: List[np.ndarray]) -> bool:
        """
        Appends a tensor to the existing tensor data with the given data ID.

        Args:
            data_id (int): The unique identifier for the tensor data.
            data (List[np.ndarray]): The tensor data to be appended.

        Returns:
            bool: True if the tensor data was successfully appended, False otherwise.

        Raises:
            ValueError: If the tensor data with the given ID does not exist in the database.
        """
        return super().append_tensor(data_id, data)
