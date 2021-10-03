import hydra
from omegaconf import DictConfig

from torch.utils.data import DataLoader

import pytorch_lightning as pl

from system import I2T

import logging
logger = logging.getLogger(__name__)

def train(cfg: DictConfig):
    train_dataset = hydra.utils.instantiate(cfg.data.train)
    val_dataset = hydra.utils.instantiate(cfg.data.val)

    logger.info(f'dataset size, train: {len(train_dataset)}, valid: {len(val_dataset)}')

    dataloader_workers = cfg.data.dataloader_workers

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size_train,
        collate_fn=train_dataset.collate_fn,
        shuffle=True,
        num_workers=dataloader_workers,
        drop_last=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size_val,
        collate_fn=val_dataset.collate_fn,
        shuffle=False,
        num_workers=dataloader_workers,
        drop_last=False
    )

    logger.info(f'dataloader size, train: {len(train_dataloader)}, valid: {len(val_dataloader)}')

    trainer = pl.Trainer(
        **cfg.train.trainer_params
    )

    model = I2T(config=cfg)

    trainer.fit(
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        model=model
    )