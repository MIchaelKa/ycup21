import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

import os
import numpy as np
from PIL import Image
from pathlib import Path
import json

from hydra.utils import instantiate

import transformers
from bpemb import BPEmb

import logging
logger = logging.getLogger(__name__)

def get_image_transform(randomize: bool):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    if randomize:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

def text_collate_fn(items):
    ids = []
    offsets = [0]
    for item in items:
        ids.append(torch.tensor(item, dtype=torch.int64))
        offsets.append(len(item))
    return {
        'ids': torch.cat(ids),
        'offsets': torch.tensor(offsets[:-1]).cumsum(dim=0)
    }

def prepare_metadata(
    metadata_directory,
    dataset_size=None,
    ):

    data = []

    # id_to_remove.npy  index_to_remove.npy
    id_to_remove = np.load(os.path.join(metadata_directory, 'id_to_remove.npy'))

    metadata_file = os.path.join(metadata_directory, 'metadata.json')

    with open(metadata_file) as json_file:
        json_strings = json_file.readlines()
    for json_string in json_strings[:dataset_size]:
        metadata = json.loads(json_string)
        image = metadata['image']
        if image not in id_to_remove:
            data.append((metadata['image'], metadata['queries']))

    logger.info(f'metadata size: {len(data)}')
        
    return data

def get_train_val(metadata, tokenizer, cfg):

    ratio = cfg.data.split_ratio
    metadata_len = len(metadata)
    train_size = int(ratio * metadata_len)

    train_metadata = metadata[:train_size]
    val_metadata = metadata[train_size:]

    train_dataset = I2TDataset(train_metadata, tokenizer, **cfg.data.train)
    val_dataset = I2TDataset(val_metadata, tokenizer, **cfg.data.val)

    logger.info(f'dataset size, train: {len(train_dataset)}, valid: {len(val_dataset)}')

    return train_dataset, val_dataset

class BERTTokenizer():
    def __init__(self, model_name):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    def __call__(self, text):

        inputs = self.tokenizer(
            text,
            # return_tensors='pt',
            # stride
            # pad_to_max_length=True,
            truncation='longest_first',
            padding='max_length',
            max_length=20
        )

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask)
        }

class BPEmbTokenizer(BPEmb):
    def __call__(self, text):
        return self.encode_ids(text)

class I2TDataset(Dataset):
    def __init__(
        self,
        metadata,
        tokenizer,
        images_directory,
        randomize = True
    ):
        super().__init__()
        self.data = metadata
        self.tokenizer = tokenizer
        self.images_directory = Path(images_directory)
        self.randomize = randomize
        self.image_transform = get_image_transform(randomize=randomize)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        img, texts = self.data[idx]
        img = Image.open((self.images_directory / f'image_{img}').with_suffix('.jpg'))
        img = img.convert('RGB')
        img = self.image_transform(img)
        if self.randomize:
            text = np.random.choice(texts)
        else:
            text = texts[0]
        # return self.tokenizer(text)
        return {'image': img, 'text': self.tokenizer(text)}
        # return {'image': img, 'text': texts}

    @staticmethod
    def collate_fn(items):
        return {
            'image': default_collate([x['image'] for x in items]),
            'text': text_collate_fn([x['text'] for x in items])
        }