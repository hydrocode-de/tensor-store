from typing import List

from .base import BaseContext

class StorageContext(BaseContext):
    def __setup_auth(self):
        # store the current JWT token
        self._anon_key = self.backend.client.supabase_key

        # add the authenticated JWT to the headers
        self.backend.client.storage._client.headers['Authorization'] = f"Bearer {self.backend._session.access_token}"

    def __restore_auth(self):
        # restore the original JWT
        self.backend.client.storage._client.headers['Authorization'] = f"Bearer {self._anon_key}"

    def _create_user_bucket(self):
        raise NotImplementedError

    def has_bucket(self) -> bool:
        raise NotImplementedError

    def list_datasets(self) -> List[str]:
        raise NotImplementedError