import unittest
from unittest.mock import MagicMock

import numpy as np

from tensorage.store import TensorStore
from tensorage.types import Dataset

class TestTensorStore(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()
    
    def test_store_init(self):
        """
        Test that the store gets initialized correctly.
        """
        pass

    def test_create_tensor_single_batch(self):
        """
        Mock the backend of the DatabaseSession and assert that the correct functions
        have been called.
        """
        # create a mock backend
        backend = MagicMock()
        
        # create the store
        store = TensorStore(backend)

        # create the dataset
        data = np.random.random((10, 10, 10))
        dataset = Dataset(13, 'test', data.shape, data.ndim, 'float32', False)
        backend.database.return_value.__enter__.return_value.insert_dataset.return_value = dataset

        # create a tensor
        store['test'] = data

        # assert that the dataset has been created
        backend.database.return_value.__enter__.return_value.insert_dataset.assert_called_once_with('test', data.shape, data.ndim)
        
        # assert that the tensor has been created
        backend.database.return_value.__enter__.return_value.insert_tensor.assert_called_once()
        
        # assert that the dataset id has been passed correctly
        assert backend.database.return_value.__enter__.return_value.insert_tensor.call_args[0][0] == dataset.id

        for expected, actual in zip(data, backend.database.return_value.__enter__.return_value.insert_tensor.call_args[0][1]):
            np.testing.assert_array_almost_equal(expected, actual)
    

if __name__ == '__main__':
    unittest.main()
