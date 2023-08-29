"""
This module defines the TensorStore class which is responsible for storing and retrieving tensor data from a Supabase backend.
The store can be accessed by using Pythons set and get item methods. Additionally, all dataset keys are also available
as attributes of the store object.
The store supports numpy-style slicing and as of now accepts and returns numpy arrays.

Example:

    .. code-block:: python
        # login
        store = login('email', 'password')

        # insert a new dataset into the store
        store['my_dataset'] = np.random.random((500, 10, 10))

        # retrieve tensor slice data from the store
        first_twelve = store.my_dataset[0:12]

        # series of subset
        subset = store['my_dataset', :, 4, 4].flatten()
"""

from typing import TYPE_CHECKING, Tuple, Union, List, Optional, Any
from typing_extensions import Literal
from dataclasses import dataclass, field
import warnings

from tqdm import tqdm
import numpy as np

from tensorage.types import Dataset

if TYPE_CHECKING:  # pragma: no cover
    from tensorage.session import BackendSession


@dataclass
class TensorStore(object):
    """
    A class representing a tensor store for storing and retrieving tensor data from a backend.

    Args:
        backend_session (BackendSession): The backend session to use for interacting with the backend.

    Attributes:
        backend (BackendSession): The backend session to use for interacting with the backend.
        quiet (bool): Whether to suppress output messages or not.
        engine (str): The engine to use for storing and retrieving tensor data.
        chunk_size (int): The chunk size to use for uploading tensor data.

    Raises:
        ValueError: If the backend session is not provided.

    """
    backend: 'BackendSession' = field(repr=False)
    quiet: bool = field(default=False)

    engine: Union[Literal['database'], Literal['storage']] = field(default='database')

    # some stuff for upload
    chunk_size: int = field(default=100000, repr=False)
    allow_overwrite: bool = False

    # add some internal metadata
    _keys: List[str] = field(default_factory=list, repr=False)

    def __post_init__(self):
        # check if the schema is installed
        with self.backend.database() as db:
            if not db.check_schema_installed():
                from tensorage.sql.sql import INIT
                SQL = INIT()
                warnings.warn(f"The schema for the TensorStore is not installed. Please connect the database and run the following script:\n\n--------8<--------\n{SQL}\n\n--------8<--------\n")
        
        # get the current keys
        self.keys()

    def get_context(self):
        raise NotImplementedError
        if self.engine == 'database':
            return self.backend.database()
        elif self.engine == 'storage':
            return self.backend.storage()
        else:
            raise ValueError(f"Unknown engine '{self.engine}'.")

    def get_select_indices(self, key: Union[str, Tuple[Union[str, slice, int]]]) -> Tuple[str, Tuple[int, int], List[Tuple[int, int]]]:
        """
        Retrieves the select indices for the given key from the database.

        Args:
            key (Union[str, Tuple[Union[str, slice, int]]]): The unique identifier for the tensor or a tuple of slice objects.

        Returns:
            Tuple[str, Tuple[int, int], List[Tuple[int, int]]]: A tuple containing the key, the shape of the tensor, and a list of index ranges to select.

        Raises:
            ValueError: If the tensor with the given key does not exist in the database.
        """        # first get key
        if isinstance(key, str):
            key = (key, )
            name = key
        elif isinstance(key[0], str):
            name = key[0]
        else:
            raise KeyError('You need to pass the key as first argument.')

        # use the Slicer
        name, index, slices = StoreSlicer(self, name)(*key[1:])
        
        # return the name, index and slices
        return name, index, slices

    def __getitem__(self, key: Union[str, Tuple[Union[str, slice, int]]]) -> np.ndarray:
        """
        Retrieves a tensor from the database with the given key or slice.

        Args:
            key (Union[str, Tuple[Union[str, slice, int]]]): The unique identifier for the tensor or a tuple of slice objects.

        Returns:
            np.ndarray: The tensor data with the given key or slice.

        Raises:
            ValueError: If the tensor with the given key does not exist in the database.
        """        
        # the user has to pass the key
        if isinstance(key, str):
            key = (key, )  #make it a tuple
            name = key
        elif isinstance(key[0], str):
            name = key[0]
        else:
            raise KeyError('You need to pass the key as first argument.')
        
        # instatiate a Slicer
        slicer = StoreSlicer(self, name)

        # for now, we accept only iloc-style indexing
        # TODO if we use the axes, we need to transform here, before instatiating the StoreSlicer
        return slicer.__getitem__(key[1:])

    def __getattr__(self, key: str) -> Any:
        """
        Instantiate a Slicer with the passed attribute used as dataset key.

        Args:
            key (str): The key of the attribute to retrieve.

        Returns:
            Any: The value of the attribute with the given key.

        Raises:
            AttributeError: If the attribute with the given key does not exist in the backend session.
        """
        # getattribute did not return anything, so now check if the key is in the keys
        if key in self._keys:
            return StoreSlicer(self, key)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def __dir__(self) -> List[str]:
        """
        Returns a list of all attributes and methods of the TensorStore object.

        Returns:
            List[str]: A list of all attributes and methods of the TensorStore object.
        """
        return super().__dir__() + self._keys

    def __setitem__(self, key: str, value: Union[List[list], np.ndarray]):
        """
        Uploads a dataset into the backend with the given key and value.

        Args:
            key (str): The unique identifier for the tensor.
            value (Union[List[list], np.ndarray]): The tensor data to be set.

        Raises:
            ValueError: If the tensor with the given key does not exist in the database.

        """        
        # check if the key is already in the database
        if key in self.keys():
            # check if we are allowed to overwrite
            if not self.allow_overwrite:
                raise ValueError(f"The key '{key}' already exists in the TensorStore. Set allow_overwrite=True to overwrite the existing dataset.")
            
            # otherwise delete the dataset
            self.__delitem__(key)

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

        # check if this should be uplaoded chunk-wise
        if value.size > self.chunk_size:
            # figure out a good batch size
            batch_size = self.chunk_size // np.multiply(*value.shape[1:])
            
            # create the index over the batch to determine the offset on upload
            single_index = np.arange(0, value.shape[0], batch_size, dtype=int)
            batch_index = list(zip(single_index, single_index[1:].tolist() + [value.shape[0]]))
            
            # build the 
            batches = [(i * batch_size, value[up:low]) for i, (up, low) in enumerate(batch_index)]
        else:
            batches = [(0, value)]
            batch_size = 1

        # connect
        with self.backend.database() as db:
            # insert the dataset
            dataset = db.insert_dataset(key, shape, dim)

            # make the iterator
            _iterator = tqdm(batches, desc=f'Uploading {key} [{len(batches)} batches of {batch_size}]') if not self.quiet else batches

            # insert the tensor
            for offset, batch in _iterator:
                db.insert_tensor(dataset.id, [tensor for tensor in batch], offset=offset)
            
            # finally update the keys
            self._keys = db.list_dataset_keys()
 
    def __delitem__(self, key: str):
        """
        Deletes a tensor from the database with the given key.

        Args:
            key (str): The unique identifier for the tensor.

        Raises:
            ValueError: If the tensor with the given key does not exist in the database.
        """
        with self.backend.database() as db:
            db.remove_dataset(key)
    
    def __contains__(self, key: str) -> bool:
        """
        Checks if a tensor with the given key exists in the database.

        Args:
            key (str): The unique identifier for the tensor.

        Returns:
            bool: True if the tensor with the given key exists in the database, False otherwise.
        """
        # get the keys
        keys = self.keys()

        # check if key is in keys
        return key in keys
    
    def __len__(self) -> int:
        """
        Returns the number of dataset keys in the database.

        Returns:
            int: The number of dataset keys in the database.
        """
        # get the keys
        keys = self.keys()

        # return the length
        return len(keys)

    def keys(self) -> List[str]:
        """
        Retrieves a list of all dataset keys in the database.

        Returns:
            List[str]: A list of all dataset keys in the database.
        """
        # get the keys from the database
        with self.backend.database() as db:
            keys = db.list_dataset_keys()
        
        # update the internal keys list
        self._keys = keys

        return keys


@dataclass
class StoreSlicer:
    """
    A class representing a slicer for a tensor store.

    Args:
        _store (TensorStore): The tensor store to slice.
        key (str): The key of the tensor to slice.
        dataset (Optional[Dataset]): The dataset to slice.

    """
    _store: TensorStore = field(repr=False)
    key: str
    dataset: Optional[Dataset] = field(default=None, repr=False)

    def __post_init__(self):
        if self.dataset is None:
            with self._store.backend.database() as db:
                self.dataset = db.get_dataset(self.key)

    def get_iloc_slices(self, *args: Union[int, Tuple[int], slice]) -> Tuple[str, Tuple[int, int], List[Tuple[int, int]]]:
        """
        Retrieves the index ranges to select from the tensor with the given key and iloc-style arguments.

        Args:
            *args (Union[int, Tuple[int], slice]): The iloc-style arguments to use for selecting the tensor data.

        Returns:
            Tuple[str, Tuple[int, int], List[Tuple[int, int]]]: A tuple containing the key, the shape of the tensor, and a list of index ranges to select.

        Raises:
            ValueError: If the tensor with the given key does not exist in the database.
        """        # check the length of the args
        if len(args) == 0:
            # use the dataset to load the full sample
            return (
                self.key, 
                [1, self.dataset.shape[0] + 1], 
                [[1, self.dataset.shape[i] + 1] for i in range(1, self.dataset.ndim)]
            )

        # slicing is actually needed
        
        # get the index
        if isinstance(args[0], int):
            index = [args[0] + 1, args[0] + 2]
        elif isinstance(args[0], slice):
            index = [args[0].start + 1, args[0].stop + 2]
        else:
            raise KeyError('Batch index needs to be passed as int or slice.')

        # get the slices
        if len(args) == 1:
            slices = [[1, self.dataset.shape[i] + 1] for i in range(2, self.dataset.ndim)]
        else:  # 2 or more beyond index
            slices = []
            for i, arg in enumerate(args[1:]):
                if isinstance(arg, int):
                    slices.append([arg + 1, arg + 1])
                elif isinstance(arg, slice):
                    slices.append([arg.start + 1 if arg.start is not None else 1, arg.stop + 1 if arg.stop is not None else self.dataset.shape[i + 1] + 1])
                else:
                    raise KeyError('Slice needs to be passed as int or slice.')
            
            # maybe the user does not want to slice all dimensions, append the others
            if len(slices) + 1 != self.dataset.ndim:   # +1 for the index
                for i in range(len(slices) + 1, self.dataset.ndim):
                    slices.append([1, self.dataset.shape[i] + 1])

        # finally return the full slice index for the database
        return (
            self.key,
            index,
            slices
        )
    
    def __getitem__(self, args: Union[int, Tuple[int], slice]) -> np.ndarray:
        """
        Retrieves a tensor from the database with the given iloc-style arguments.

        Args:
            args (Union[int, Tuple[int], slice]): The iloc-style arguments to use for selecting the tensor data.

        Returns:
            np.ndarray: The tensor data with the given iloc-style arguments.

        Raises:
            ValueError: If the tensor with the given key does not exist in the database.
        """        # get the slices
        _, index, slices = self.get_iloc_slices(*args)

        # load the tensor
        with self._store.backend.database() as db:
            # load the tensor
            arr = db.get_tensor(self.key, index[0], index[1], [s[0] for s in slices], [s[1] for s in slices])
        
        # TODO now we can transform to other libaries
        return arr

    def __call__(self, *args: Union[int, Tuple[int], slice]) -> Tuple[str, Tuple[int, int], List[Tuple[int, int]]]:
        """
        Retrieves the index ranges to select from the tensor with the given iloc-style arguments.

        Args:
            *args (Union[int, Tuple[int], slice]): The iloc-style arguments to use for selecting the tensor data.

        Returns:
            Tuple[str, Tuple[int, int], List[Tuple[int, int]]]: A tuple containing the key, the shape of the tensor, and a list of index ranges to select.

        Raises:
            ValueError: If the tensor with the given key does not exist in the database.
        """
        return self.get_iloc_slices(*args)
