import hydra
from omegaconf import DictConfig

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from system import I2T
from dataset import prepare_metadata, get_train_val

from hydra.utils import instantiate

import logging
logger = logging.getLogger(__name__)

def train(cfg: DictConfig):
    seed_everything(42, workers=True)

    metadata = prepare_metadata(**cfg.data.metadata)

    tokenizer = instantiate(cfg.tokenizer)

    train_dataset, val_dataset = get_train_val(metadata, tokenizer, cfg)

    dataloader_workers = cfg.data.dataloader_workers

    if cfg.data.collate_fn:
        collate_fn = train_dataset.collate_fn
    else:
        collate_fn = None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size_train,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=dataloader_workers,
        drop_last=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size_val,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=dataloader_workers,
        drop_last=False
    )

    logger.info(f'dataloader size, train: {len(train_dataloader)}, valid: {len(val_dataloader)}')

    early_stopping = EarlyStopping(
        monitor='val/nce',
        patience=3,
        verbose=True,
        mode='min',
        strict=True
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val/nce',
        mode='min'
    )

    callbacks=[early_stopping]

    if cfg.train.trainer_params.checkpoint_callback:
        callbacks.append(checkpoint_callback)

    trainer = pl.Trainer(
        **cfg.train.trainer_params,
        callbacks=callbacks
    )

    # pass tokenizer
    model = I2T(config=cfg)

    trainer.fit(
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        model=model
    )