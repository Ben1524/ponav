from typing import Optional

from .dsrnn import DSRNNPolicy


class OMGRUPolicy(DSRNNPolicy):
    def __init__(self, checkpoint_path: Optional[str] = None):
        super().__init__(policy_name='om_gru', checkpoint_path=checkpoint_path)
