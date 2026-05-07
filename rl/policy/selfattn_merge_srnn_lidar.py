from typing import Optional

from .dsrnn import DSRNNPolicy


class SelfAttnMergeSRNNLidarPolicy(DSRNNPolicy):
    def __init__(self, checkpoint_path: Optional[str] = None):
        super().__init__(policy_name='selfAttn_merge_srnn_lidar', checkpoint_path=checkpoint_path)
