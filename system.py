import pytorch_lightning as pl
import torch
from torch import nn

from hydra.utils import instantiate
from omegaconf import DictConfig

from typing import List, Dict, Tuple

class I2T(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters(config)
        self.modalities = ('text', 'image')
        self.encoders = nn.ModuleDict({
            modality: instantiate(self.hparams.model.get(modality))
            for modality in self.modalities
        })
        # hard-coded ntxent loss for simplicity
        self.loss = nn.CrossEntropyLoss()

    def forward(self, batch: Dict) -> Dict:
        return {
            modality: self.encoders[modality](batch[modality])
            for modality in self.modalities
        }

    def training_step(self, batch: Dict, batch_idx: int) -> Dict:
        return self.step_model(self(batch), mode='train')

    def validation_step(self, batch: Dict, batch_idx: int) -> Dict:
        return self.step_model(self(batch), mode='val')

    # def validation_epoch_end(self, batch):
    #     print(batch)

    def step_model(self, local_outputs: Dict[str, torch.Tensor], mode: str) -> Dict:
        temperature = 0.01
        logits = self.gather_logits(local_outputs) / temperature
        losses = self.calculate_loss(logits)
        metrics = self.calculate_metrics(logits)
        # on_step=True, on_epoch=True
        self.log_dict({f'{mode}/{name}': value for name, value in losses.items()})
        self.log_dict({f'{mode}/{name}': value for name, value in metrics.items()})
        return {'loss': losses['nce']}

    def gather_logits(self, local_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        image_features = local_outputs['image']
        text_features = local_outputs['text']
        return image_features @ text_features.T

    def calculate_loss(self, logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Contrastive NCE loss, see https://paperswithcode.com/method/nt-xent for details
        """
        labels = torch.arange(0, logits.shape[0], device=self.device)
        loss_i2t = self.loss(logits, labels)
        loss_t2i = self.loss(logits.T, labels)
        loss = (loss_i2t + loss_t2i) / 2
        return {
            'nce_i2t': loss_i2t,
            'nce_t2i': loss_t2i,
            'nce': loss
        }

    def calculate_metrics(self, logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            'binary_accuracy': (logits.diag().unsqueeze(1) >= logits).to(dtype=torch.float32).mean()
        }

    def configure_optimizers(self):
        learning_rate = 1e-4
        return torch.optim.Adam(self.parameters(), lr=learning_rate)
