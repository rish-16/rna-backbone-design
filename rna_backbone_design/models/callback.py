import torch
from pytorch_lightning import Callback

class NanGradientCallback(Callback):
    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        """Called right before an optimizer step to check gradients for NaNs."""
        if self._any_nan_gradients(pl_module):
            trainer.logger.log_metrics({"nan_gradient_detected": 1}, step=trainer.global_step)
            pl_module.zero_grad(set_to_none=True)  # Clear gradients
            print(f"NaN gradients detected at global step {trainer.global_step}, skipping optimizer step")

    def _any_nan_gradients(self, pl_module):
        """Check if any gradient is NaN in the model"""
        for param in pl_module.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                return True
        return False