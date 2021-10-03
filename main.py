import hydra
from omegaconf import DictConfig, OmegaConf
import os

from train import train

from hydra.utils import instantiate

import logging
logger = logging.getLogger(__name__)

@hydra.main(config_path="config", config_name="base")
def main(cfg: DictConfig) -> None:

    # print(OmegaConf.to_yaml(cfg))

    # test_data(cfg)
    # test_model(cfg)

    train(cfg)
    logger.info("output directory : {}".format(os.getcwd()))

def test_model(cfg: DictConfig):

    model = instantiate(cfg.model.image)
    print(model)


def test_data(cfg: DictConfig):

    dataset = hydra.utils.instantiate(cfg.data.train)
    # dataset = I2TDataset(**cfg.data.train)
    print(len(dataset))

    dataset_item = dataset[0]

    image = dataset_item['image']
    print(image.shape)

    print(dataset_item['text'])
    

if __name__ == "__main__":
    main()