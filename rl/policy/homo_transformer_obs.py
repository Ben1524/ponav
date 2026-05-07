from typing import Optional

from .dsrnn import DSRNNPolicy


class HomoTransformerObsPolicy(DSRNNPolicy):
    def __init__(self, checkpoint_path: Optional[str] = None):
        super().__init__(policy_name='homo_transformer_obs', checkpoint_path=checkpoint_path)
