# Importing Libraries
import argparse

import yaml
#from aim import Run

import wandb

from dataloader import load_dataloader
from trainer import Trainer
from transformer import VQGANTransformer
from vqgan import VQGAN


def main(args, config):

    vqgan = VQGAN(**config["architecture"]["vqgan"])
    transformer = VQGANTransformer(
        vqgan, **config["architecture"]["transformer"], device=args.device
    )
    dataloader = load_dataloader(name=args.dataset_name, batch_size=args.mini_batch_size)
    update_every = args.batch_size // args.mini_batch_size

    run = wandb.init(entity="wqlevi", project=args.dataset_name, group="vqgan", config=config)
    #run["hparams"] = config

    trainer = Trainer(
        vqgan,
        transformer,
        run=run,
        name=args.dataset_name,
        config=config["trainer"],
        seed=args.seed,
        device=args.device,
        experiment_dir='debug_experiments'
    )

    trainer.train_vqgan(dataloader, epochs=50, update_every = update_every)
    trainer.train_transformers(dataloader, epochs=50, update_every = update_every)
    trainer.generate_images()

    wandb.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        default="configs/default.yml",
        help="path to config file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for training",
    )
    parser.add_argument(
        "--mini-batch-size",
        type=int,
        default=8,
        help="Mini batch size for training",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        choices=["mnist", "cifar", "custom", "moco"],
        default="mnist",
        help="Dataset for the model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device to train the model on",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="Device ID to use for training (if using multiple GPUs)",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default=42,
        help="Seed for Reproducibility",
    )

    args = parser.parse_args()

    args.device = args.device + ":" + str(args.device_id) if args.device == "cuda" else "" # completion on specific device


    with open(args.config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(args, config)
