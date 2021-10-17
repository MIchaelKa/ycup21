import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F

from hydra.utils import instantiate
from omegaconf import DictConfig

from typing import List, Dict, Tuple

import logging
logger = logging.getLogger(__name__)

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

    def on_epoch_start(self):
        print('\n')

    def validation_epoch_end(self, batch):
        # print(batch)
        print('\n')

    # call insted of log_dict and disable progress bar?
    def my_log_dict(self, mode, dict):
        self.log_dict({f'{mode}/{name}': value for name, value in dict.items()}, on_step=True, on_epoch=True)
        logger.info(f'my_log_dict: {dict}')

    def kl_div_loss(self, z_mu, z_log_var):
        kl_div = -0.5 * torch.sum(1 + z_log_var - z_mu**2 - torch.exp(z_log_var), axis=1) # sum over latent dimension
        kl_div = kl_div.mean() # average over batch dimension

        # kl_loss = (-0.5 * (1 + z_log_var - z_mu**2 - torch.exp(z_log_var)).sum(dim = 1)).mean(dim = 0)
        
        return kl_div

    def step_model_vae(self, local_outputs: Dict[str, torch.Tensor], mode: str) -> Dict:
        temperature = 0.01

        image_features, image_z_mean, image_z_log_var = local_outputs['image']
        text_features, text_z_mean, text_z_log_var = local_outputs['text']

        logits = (image_features @ text_features.T) / temperature

        nce_losses = self.calculate_loss(logits)
        metrics = self.calculate_metrics(logits)

        image_kl_loss = self.kl_div_loss(image_z_mean, image_z_log_var)
        text_kl_loss = self.kl_div_loss(text_z_mean, text_z_log_var)

        alpha = 0.95

        loss = alpha * nce_losses['nce'] + (1 - alpha) * (image_kl_loss + text_kl_loss)

        # print(nce_losses['nce'], image_kl_loss, text_kl_loss)

        self.log_dict({f'{mode}/{name}': value for name, value in nce_losses.items()})
        self.log_dict({f'{mode}/{name}': value for name, value in metrics.items()})
        self.log(f'{mode}/image_kl_loss', image_kl_loss)
        self.log(f'{mode}/text_kl_loss', image_kl_loss)
        self.log(f'{mode}/loss', loss)

        return {'loss': loss}

    def step_model_distance(self, local_outputs: Dict[str, torch.Tensor], mode: str) -> Dict:
        image_features = local_outputs['image']
        text_features = local_outputs['text']
        logits = image_features @ text_features.T
        return logits.diag()

    def step_model_arc_face(self, local_outputs: Dict[str, torch.Tensor], mode: str) -> Dict:
        m=0.4

        logits = self.gather_logits(local_outputs)
        arcosine = logits.arccos()
        labels = torch.arange(0, logits.shape[0], device=self.device)
        arcosine += F.one_hot(labels, num_classes=logits.shape[0]) * m
        logits = arcosine.cos()

        losses = self.calculate_loss(logits)
        metrics = self.calculate_metrics(logits)
        # on_step=True, on_epoch=True
        self.log_dict({f'{mode}/{name}': value for name, value in losses.items()})
        self.log_dict({f'{mode}/{name}': value for name, value in metrics.items()})
        return {'loss': losses['nce']}

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
        
        return torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=learning_rate)
        # return torch.optim.Adam(self.parameters(), lr=learning_rate)
