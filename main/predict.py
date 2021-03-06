# hydra imports
from omegaconf import OmegaConf

# generic imports
from typing import Optional, List, Iterable
from tqdm.auto import tqdm
from PIL import Image
import numpy as np
import more_itertools
import os
import json
import click

# torch imports
import torch
from torch.utils.data._utils.collate import default_collate

# custom imports
from system import I2T
from dataset import get_image_transform, text_collate_fn

from hydra.utils import instantiate

class I2TInferer(object):
    def __init__(
        self,
        ckpt_path: str,
        device: str
    ):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        cfg = OmegaConf.create(ckpt['hyper_parameters'])
        print(cfg.tokenizer)
        self.tokenizer = instantiate(cfg.tokenizer)
        self.image_transform = get_image_transform(randomize=False)
        model = I2T(config=cfg)
        model = model.eval()
        model.load_state_dict(ckpt['state_dict'])
        model = model.to(device)
        self.model = model
        self.text_collate_fn = text_collate_fn
        self.img_collate_fn = default_collate
        self.device = device
        
    # def encode_texts(self, texts: Iterable[str]) -> torch.Tensor:
    #     ids = [self.tokenizer.encode_ids(text) for text in texts]
    #     texts_torch = self.text_collate_fn(ids)
    #     texts_torch['ids'] = texts_torch['ids'].to(self.device)
    #     texts_torch['offsets'] = texts_torch['offsets'].to(self.device)
    #     return self.model.encoders['text'](texts_torch).cpu().detach()

    def encode_texts(self, texts):
        ids = [self.tokenizer(text) for text in texts]
        texts_torch = default_collate(ids)
        texts_torch['input_ids'] = texts_torch['input_ids'].to(self.device)
        texts_torch['attention_mask'] = texts_torch['attention_mask'].to(self.device)    
        return self.model.encoders['text'].encode(texts_torch).cpu().detach()
    
    def encode_images(self, images: Iterable[Image.Image]) -> torch.Tensor:
        pbar = tqdm()
        image_features = []
        for chunk in more_itertools.chunked(images, 10):
            images = [self.image_transform(x.convert('RGB')) for x in chunk]
            images_torch = self.img_collate_fn(images).to(self.device)
            chunk_image_features = self.model.encoders['image'].encode(images_torch).cpu().detach()
            image_features.append(chunk_image_features)
            pbar.update(len(chunk))
        pbar.close()
        return torch.cat(image_features, dim=0)
    
    def predict(self, images: Iterable[Image.Image], classes: Iterable[str]) -> np.ndarray:
        text_features = self.encode_texts(classes)
        image_features = self.encode_images(images)
        logits = image_features @ text_features.T
        return torch.argmax(logits, dim=1).numpy()


@click.command()
@click.option('--ckpt_path', help='Path to PL checkpoint')
@click.option('--data_directory', help='Path to directory with evaluation datasets')
@click.option('--predicts_file', help='Path to file where predictions should be put to')
@click.option('--limit_samples', default=None, type=int, help='Limit num of evaluated images')
@click.option('--device', default='cpu', help='PyTorch device')
@click.option('--num_threads', default=None, type=int, help='Optionally force number of torch threads')
@click.option('--dataset', '-d', default=None, multiple=True, help='Optionally select datasets manually')
@torch.no_grad()
def main(
    ckpt_path: str,
    data_directory: str,
    predicts_file: str,
    limit_samples: Optional[int],
    device: str,
    num_threads: Optional[int],
    dataset: Optional[List[str]]
):
    if num_threads is not None:
        torch.set_num_threads(num_threads)

    inferer = I2TInferer(ckpt_path=ckpt_path, device=device)
    if dataset:
        datasets = dataset
    else:
        datasets = os.listdir(data_directory)
        # datasets.remove('.DS_Store')
        # datasets.remove('._.DS_Store')    

    results = {}
    for dataset in datasets:
        with open(f"{data_directory}/{dataset}/classes.json", 'r') as f:
            classes_labels = json.load(f)
        image_files = os.listdir(f'{data_directory}/{dataset}/img')
        if limit_samples is not None:
            image_files = image_files[:limit_samples]
        images = (Image.open(f'{data_directory}/{dataset}/img/{file}') for file in image_files)
        predicts = inferer.predict(images, classes_labels).tolist()
        predicts = {file.split('.')[0]: predict for file, predict in zip(image_files, predicts)}
        results[dataset] = predicts
    with open(predicts_file, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
