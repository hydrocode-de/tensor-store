from typing import TYPE_CHECKING, Tuple, Union, List, Optional, Any
from typing_extensions import Literal
from dataclasses import dataclass, field
import warnings

from tqdm import tqdm
import numpy as np

from tensorage.types import Dataset

if TYPE_CHECKING:
    from tensorage.session import BackendSession


@dataclass
class TensorStore(object):
    backend: 'BackendSession' = field(repr=False)
    quiet: bool = field(default=False)

    engine: Union[Literal['database'], Literal['storage']] = field(default='database')

    # some stuff for upload
    chunk_size: int = field(default=100000, repr=False)

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
        # first get key
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

    def __getitem__(self, key: Union[str, Tuple[Union[str, slice, int]]]):
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

    def __getattr__(self, key: str):
        # getattribute did not return anything, so now check if the key is in the keys
        if key in self._keys:
            return StoreSlicer(self, key)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def __dir__(self):
        return super().__dir__() + self._keys

    def __setitem__(self, key: str, value: Union[List[list], np.ndarray]):
        # check if the key is already in the database
        if key in self.keys():
            # TODO here we need to update the dataset in the database
            raise NotImplementedError('Updating datasets is not implemented yet.')

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
            if batch_size == 0:
                batch_size = 1
            
            # create the index over the batch to determine the offset on upload
            single_index = np.arange(0, value.shape[0], batch_size, dtype=int)
            batch_index = list(zip(single_index, single_index[1:].tolist() + [value.shape[0]]))
            
            # build the 
            batches = [(i * batch_size, value[up:low]) for i, (up, low) in enumerate(batch_index)]
        else:
            batches = [(0, value)]

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
        with self.backend.database() as db:
            db.remove_dataset(key)
    
    def __contains__(self, key: str):
        # get the keys
        keys = self.keys()

        # check if key is in keys
        return key in keys
    
    def __len__(self):
        # get the keys
        keys = self.keys()

        # return the length
        return len(keys)

    def keys(self) -> List[str]:
        # get the keys from the database
        with self.backend.database() as db:
            keys = db.list_dataset_keys()
        
        # update the internal keys list
        self._keys = keys

        return keys


@dataclass
class StoreSlicer:
    _store: TensorStore = field(repr=False)
    key: str
    dataset: Optional[Dataset] = field(default=None, repr=False)

    def __post_init__(self):
        if self.dataset is None:
            with self._store.backend.database() as db:
                self.dataset = db.get_dataset(self.key)

    def get_iloc_slices(self, *args: Union[int, Tuple[int], slice]) -> Tuple[str, Tuple[int, int], List[Tuple[int, int]]]:
        # check the length of the args
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
        # get the slices
        _, index, slices = self.get_iloc_slices(*args)

        # load the tensor
        with self._store.backend.database() as db:
            # load the tensor
            arr = db.get_tensor(self.key, index[0], index[1], [s[0] for s in slices], [s[1] for s in slices])
        
        # TODO now we can transform to other libaries
        return arr

    def __call__(self, *args: Union[int, Tuple[int], slice]) -> Tuple[str, Tuple[int, int], List[Tuple[int, int]]]:
        return self.get_iloc_slices(*args)
