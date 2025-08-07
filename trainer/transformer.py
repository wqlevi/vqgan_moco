# Importing Libraries
import torchvision

import torch
import torch.nn as nn
import torch.nn.functional as F
from wandb import Image

from utils import count_parameters


class TransformerTrainer:
    def __init__(
        self,
        model: nn.Module,
        run,
        experiment_dir: str = "experiments",
        device: str = "cuda",
        learning_rate: float = 4.5e-06,
        beta1: float = 0.9,
        beta2: float = 0.95,
    ):
        self.run = run
        self.experiment_dir = experiment_dir
        self.global_step = 0

        self.model = model
        self.device = device
        self.optim = self.configure_optimizers(
            learning_rate=learning_rate, beta1=beta1, beta2=beta2
        )

    def configure_optimizers(
        self, learning_rate: float = 4.5e-06, beta1: float = 0.9, beta2: float = 0.95
    ):
        decay, no_decay = set(), set()
        whitelist_weight_modules = (nn.Linear,)
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

        # Enabling weight decay to only certain layers
        for mn, m in self.model.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn

                if pn.endswith("bias"):
                    no_decay.add(fpn)

                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)

                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        no_decay.add("pos_emb")

        param_dict = {pn: p for pn, p in self.model.transformer.named_parameters()}

        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": 0.01,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(beta1, beta2)
        )
        return optimizer

    def train(self, dataloader: torch.utils.data.DataLoader, epochs: int):
        print(f"Training Transformer with {count_parameters(self.model)/(1024**2)} M parameters")
        for epoch in range(epochs):
            for index, imgs in enumerate(dataloader):
                self.global_step += 1
                self.optim.zero_grad()
                imgs = imgs.to(device=self.device)
                logits, targets = self.model(imgs)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
                )
                loss.backward()
                self.optim.step()

                self.run.log(
                    {"Cross Entropy Loss":
                    loss,
                    },
                    step=self.global_step,
                )

                if index % 10 == 0:
                    print(
                        f"Epoch: {epoch+1}/{epochs} | Batch: {index}/{len(dataloader)} | Cross Entropy Loss : {loss:.4f}"
                    )

                    _, sampled_imgs = self.model.log_images(imgs[0][None])

                    self.run.log(
                        {
                            "Transformer Images":
                            Image(
                            torchvision.utils.make_grid(sampled_imgs, normalize=True)
                            .permute(1, 2, 0)
                            .cpu()
                            #.mul(255)
                            #.add_(0.5)
                            #.clamp_(0, 255)
                            #.permute(1, 2, 0)
                            #.to("cpu", torch.uint8)
                            .numpy(), 
                            caption= "Transformer Images",
                        ),
                        },
                        step=self.global_step,
                    )
