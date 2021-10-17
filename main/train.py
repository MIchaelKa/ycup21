from omegaconf import DictConfig

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, BaseFinetuning

from system import I2T
from dataset import prepare_metadata, get_train_val

from hydra.utils import instantiate

import logging
logger = logging.getLogger(__name__)

import torch

class FeatureExtractorFreezeUnfreeze(BaseFinetuning):

    def __init__(self, unfreeze_at_epoch=5):
        self._unfreeze_at_epoch = unfreeze_at_epoch

    def freeze_before_training(self, pl_module):
        logger.info(f'freeze_before_training')
        self.freeze(pl_module.encoders.image)

    def finetune_function(self, pl_module, current_epoch, optimizer, optimizer_idx):
        logger.info(f'finetune_function: {current_epoch}')
        # When `current_epoch` is 10, feature_extractor will start training.
        if current_epoch == self._unfreeze_at_epoch:
            self.unfreeze_and_add_param_group(
                modules=pl_module.encoders.image,
                optimizer=optimizer,
                train_bn=True,
            )

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
        monitor=cfg.train.monitor,
        patience=3,
        verbose=True,
        mode='min',
        strict=True
    )

    # freeze = FeatureExtractorFreezeUnfreeze()

    callbacks=[early_stopping]

    if cfg.train.trainer_params.checkpoint_callback:
        checkpoint_callback = instantiate(cfg.train.checkpoint_callback)
        callbacks.append(checkpoint_callback)

    trainer = pl.Trainer(
        **cfg.train.trainer_params,
        callbacks=callbacks
    )

    model = I2T(config=cfg)

    ckpt_path = '/content/drive/MyDrive/ai/ycup21/checkpoints/baseline_image_encoder.ckpt'
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.encoders.image.load_state_dict(ckpt)

    for param in model.encoders.image.parameters():
        param.requires_grad = False

    trainer.fit(
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        model=model
    )