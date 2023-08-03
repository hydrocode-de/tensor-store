from typing import TYPE_CHECKING, Tuple, Union, List
from dataclasses import dataclass, field
import warnings

from tqdm import tqdm
import numpy as np

if TYPE_CHECKING:
    from tensorage.session import BackendSession


@dataclass
class TensorStore(object):
    backend: 'BackendSession' = field(repr=False)
    quiet: bool = field(default=False)

    # some stuff for upload
    chunk_size: int = field(default=100000, repr=False)

    def __post_init__(self):
        # check if the schema is installed
        with self.backend.database() as db:
            if not db.check_schema_installed():
                from tensorage.db_init import SQL
                warnings.warn(f"The schema for the TensorStore is not installed. Please connect the database and run the following script:\n\n--------8<--------\n{SQL}\n\n--------8<--------\n")

    def __getitem__(self, key: Union[str, Tuple[Union[str, slice, int]]]):
        # first get key
        if isinstance(key, str):
            name = key
        elif isinstance(key[0], str):
            name = key[0]
        else:
            raise KeyError('You need to pass the key as first argument.')
        
        # load the dataset
        with self.backend.database() as db:
            dataset = db.get_dataset(name)

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
        with self.backend.database() as db:
            # load the tensor
            arr = db.get_tensor(name, index[0], index[1], [s[0] for s in slices], [s[1] for s in slices])
        
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
        
        # TODO: here caching could be added
        return keys
