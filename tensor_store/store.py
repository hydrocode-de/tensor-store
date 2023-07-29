from typing import TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from tensor_store.session import BackendSession


@dataclass
class TensorStore(object):
    _session: 'BackendSession' = field(repr=False)
