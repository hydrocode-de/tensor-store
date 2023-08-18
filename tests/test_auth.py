import unittest
from unittest.mock import Mock, patch, mock_open
import json

from tensorage.auth import link_to, login, signup, SUPA_FILE, _get_auth_info
from tensorage.store import TensorStore

backend_config = dict(SUPABASE_URL='https://test.com', SUPABASE_KEY='test_key')
backend_config_json = json.dumps(backend_config)


class TestBackendSession(unittest.TestCase):
    @patch('builtins.open', new_callable=mock_open())
    def test_link_backend(self, m):
        """
        Test that the link_to function creates the SUPA_FILE file.
        """
        return_value = link_to('https://test.com', 'test_key')

        # assert that the file was written
        m.assert_called_once_with(SUPA_FILE, 'w')

        # assert that the return value is True
        self.assertTrue(return_value)
    
    @patch('tensorage.backend.database.DatabaseContext.check_schema_installed', return_value=True)
    @patch('builtins.open', new_callable=mock_open(read_data=backend_config_json))
    @patch('tensorage.session.BackendSession.client')
    def test_link_backend_and_login(self, client, m, db):
        """
        Test that the link_to function creates the SUPA_FILE file and logs in.
        """
        auth_response = Mock()
        auth_response.user = dict(username='test_username')
        auth_response.session = dict(access_token='test_access_token', refresh_token='test_refresh_token')
        client.auth.sign_in_with_password.return_value = auth_response

        # mock link the with login
        with patch('tensorage.backend.database.DatabaseContext.list_dataset_keys', return_value=['foo', 'bar']) as d:
            store = link_to('https://test.com', 'test_key', 'test_email', 'test_password')

        # assert that the file was written
        m.assert_called_with(SUPA_FILE, 'w')

        # assert that the login was called
        self.assertEqual(store.backend._user, auth_response.user)
        self.assertEqual(store.backend._session, auth_response.session)

        # assert that the store is of type TensorStore
        self.assertIsInstance(store, TensorStore)

    def test_login_execption(self):
        """Test that the missing password execption is raised."""
        # catch the RuntimeError
        with self.assertRaises(RuntimeError) as err:
            login('test_email', None, 'https://test.com', 'test_key')

        # make sure its the right error message
        self.assertTrue('Email and password are not saved in' in str(err.exception))

    def test_missing_key(self):
        """Test the missing SUPABASE_KEY execption."""
        with self.assertRaises(RuntimeError) as err:
            _get_auth_info(backend_url='https://test.com', backend_key=None)
        
        # make sure its the right error message
        self.assertTrue('SUPABASE_KEY environment variable' in str(err.exception))

    @patch('builtins.open', new_callable=mock_open(read_data=backend_config_json))
    @patch('tensorage.session.BackendSession.client')
    def test_signup(self, client, fs):
        """
        Test the register function of the backend session
        """
        # mock the sign_up function
        auth_response = Mock()
        auth_response.user = dict(username='test_email')
        auth_response.session = dict(access_token='test_access_token', refresh_token='test_refresh_token')
        client.auth.sign_up.return_value = auth_response

        # test the signup function
        response = signup('test_email', 'test_password')
        
        # assert that the response is correct
        self.assertEqual(response.user, auth_response.user)

