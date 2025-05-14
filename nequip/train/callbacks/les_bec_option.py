import logging
from lightning.pytorch.callbacks import Callback
from nequip.nn.les_energy import LatentEwaldSum
from typing import Optional

class ToggleLESCallback(Callback):
    """Callback to toggle the compute_bec option in LatentEwaldSum during testing."""
    def __init__(self, compute_bec: bool = True, bec_output_index: Optional[int] = None):
        super().__init__()
        self.compute_bec = compute_bec
        self.bec_output_index = bec_output_index
        self.logger = logging.getLogger(__name__)
    def on_test_start(self, trainer, pl_module):
        found = False
        for m in pl_module.model.modules():
            if isinstance(m, LatentEwaldSum):
                found = True
                m.compute_bec = self.compute_bec
                self.logger.info(f"Setting compute_bec to {self.compute_bec} in LatentEwaldSum")
                if self.bec_output_index is not None:
                    m.bec_output_index = self.bec_output_index
                    self.logger.info(f"Setting bec_output_index to {self.bec_output_index}")
        if not found:
            self.logger.warning("LatentEwaldSum module not found in the model.")

    # def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
    #     """Called when a test batch ends to check if BEC is in the output."""
    #     od = outputs[f"test_{dataloader_idx}_output"].copy()
    #     if "BEC" in od:
    #         self.logger.info(f"BEC shape: {od['BEC'].shape}")
    #     else:
    #         self.logger.warning("BEC not found in the output.")