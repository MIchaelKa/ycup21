import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

import numpy as np
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
import jsonlines
import json

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

class I2TDataset(Dataset):
    def __init__(
        self,
        metadata_file,
        images_directory,
        tokenizer,
        read_jsonlines = True,
        start = 0,
        end = None,
        randomize = True,
        tqdm_load = False
    ):
        super().__init__()
        self.data = []

        skip_images = [13,44]

        if read_jsonlines:
            with jsonlines.open(metadata_file) as reader:
                if tqdm_load:
                    reader = tqdm(reader)
                for obj in reader:
                    self.data.append((obj['image'], obj['queries']))
            self.data = self.data[slice(start, end)]
        else:
            with open(metadata_file) as json_file:
                json_strings = json_file.readlines()
            for json_string in json_strings[start:end]:
                metadata = json.loads(json_string)
                image = metadata['image']
                if image not in skip_images:
                    self.data.append((metadata['image'], metadata['queries']))            

        self.images_directory = Path(images_directory)
        self.randomize = randomize
        self.image_transform = get_image_transform(randomize=randomize)
        self.tokenizer = tokenizer

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
        return {'image': img, 'text': self.tokenizer.encode_ids(text)}

    @staticmethod
    def collate_fn(items):
        return {
            'image': default_collate([x['image'] for x in items]),
            'text': text_collate_fn([x['text'] for x in items])
        }