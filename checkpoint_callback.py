# checkpoint_callback.py
import os
from transformers import TrainerCallback

class SaveBestModelCallback(TrainerCallback):
    def __init__(self, save_path):
        self.best_metric = None
        self.save_path = save_path

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric = metrics.get("eval_loss")
        if self.best_metric is None or metric < self.best_metric:
            self.best_metric = metric
            # Save model checkpoint
            kwargs["model"].save_pretrained(self.save_path)
            print(f"New best model saved with eval_loss: {self.best_metric}")

# Usage in training:
# trainer.add_callback(SaveBestModelCallback(save_path="./best_checkpoint"))
