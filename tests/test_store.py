import unittest
from unittest.mock import MagicMock
import warnings

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
        # create a mock backend
        backend = MagicMock()

        # create the store
        store = TensorStore(backend)
        
        # make sure schema has been checked
        backend.database.return_value.__enter__.return_value.check_schema_installed.assert_called_once()

    def test_store_init_raises_warning(self):
        """
        Test that the store raises a warning if the schema is not installed.
        """
        # create a mock backend
        backend = MagicMock()

        # mock the schema installed method
        backend.database.return_value.__enter__.return_value.check_schema_installed.return_value = False

        # create the store and catch the warning
        with warnings.catch_warnings(record=True) as w:
            store = TensorStore(backend)
            
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
            assert "The schema for the TensorStore is not installed." in str(w[-1].message)

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
    
    def test_create_tensor_multiple_batches(self):
        """
        Mock the backend of the DatabaseSession and assert that the correct functions
        have been called.
        """
        # create a mock backend
        backend = MagicMock()

        # create the store
        store = TensorStore(backend)

        # create the dataset
        data = np.random.random((30, 10, 10))
        dataset = Dataset(14, 'test2', data.shape, data.ndim, 'float32', False)
        backend.database.return_value.__enter__.return_value.insert_dataset.return_value = dataset

        # set a smaller chunk size to simulate multiple batches
        store.chunk_size = data.shape[1] * data.shape[2] * 2  # data.shape[0] // 2 is batch size
        
        # create the tensor
        store['test2'] = data

        # assert that the dataset has been created
        backend.database.return_value.__enter__.return_value.insert_dataset.assert_called_once_with('test2', data.shape, data.ndim)

        # assert that the tensor has been created
        expected_batch_size = int(np.ceil(data.shape[0] / 2))
        assert backend.database.return_value.__enter__.return_value.insert_tensor.call_count == expected_batch_size


if __name__ == '__main__':
    unittest.main()
