"""
https://github.com/dome272/VQGAN-pytorch/blob/main/training_vqgan.py
"""

# Importing Libraries
import os
import ipdb

import imageio
import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
#from aim import Image, Run

from wandb import Image

from utils import weights_init, count_parameters
from vqgan import Discriminator


class VQGANTrainer:
    """Trainer class for VQGAN, contains step, train methods"""

    def __init__(
        self,
        model: torch.nn.Module,
        run,
        # Training parameters
        device: str or torch.device = "cuda",
        learning_rate: float = 2.25e-05,
        beta1: float = 0.5,
        beta2: float = 0.9,
        # Loss parameters
        perceptual_loss_factor: float = 1.0,
        rec_loss_factor: float = 1.0,
        # Discriminator parameters
        disc_factor: float = 1.0,
        disc_start: int = 100,
        # Miscellaneous parameters
        experiment_dir: str = "./experiments",
        perceptual_model: str = "vgg",
        save_every: int = 10,
        update_every: int = 1,
    ):

        self.run = run
        self.device = device

        # VQGAN parameters
        self.vqgan = model

        # Discriminator parameters
        self.discriminator = Discriminator(image_channels=self.vqgan.img_channels).to(
            self.device
        )
        self.discriminator.apply(weights_init)

        # Loss parameters
        self.perceptual_loss = lpips.LPIPS(net=perceptual_model).to(self.device)

        # Optimizers
        self.opt_vq, self.opt_disc = self.configure_optimizers(
            learning_rate=learning_rate, beta1=beta1, beta2=beta2
        )

        # Hyperprameters
        self.disc_factor = disc_factor
        self.disc_start = disc_start
        self.perceptual_loss_factor = perceptual_loss_factor
        self.rec_loss_factor = rec_loss_factor

        # Save directory
        self.expriment_save_dir = experiment_dir

        # Miscellaneous
        self.global_step = 0
        self.sample_batch = None
        self.gif_images = []
        self.save_every = save_every
        self.update_every = update_every

    def configure_optimizers(
        self, learning_rate: float = 2.25e-05, beta1: float = 0.5, beta2: float = 0.9
    ):
        opt_vq = torch.optim.Adam(
            list(self.vqgan.encoder.parameters())
            + list(self.vqgan.decoder.parameters())
            + list(self.vqgan.codebook.parameters())
            + list(self.vqgan.quant_conv.parameters())
            + list(self.vqgan.post_quant_conv.parameters()),
            lr=learning_rate,
            eps=1e-08,
            betas=(beta1, beta2),
        )
        opt_disc = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=learning_rate,
            eps=1e-08,
            betas=(beta1, beta2),
        )

        return opt_vq, opt_disc

    def step(self, idx:int, imgs: torch.Tensor) -> torch.Tensor:
        """Performs a single training step from the dataloader images batch

        For the VQGAN, it calculates the perceptual loss, reconstruction loss, and the codebook loss and does the backward pass.

        For the discriminator, it calculates lambda for the discriminator loss and does the backward pass.

        Args:
            idx: current index of the step
            imgs: input tensor of shape (batch_size, channel, H, W)

        Returns:
            decoded_imgs: output tensor of shape (batch_size, channel, H, W)
        """

        # Getting decoder output
        decoded_images, _, q_loss = self.vqgan(imgs)

        """
        =======================================================================================================================
        VQ Loss
        """
        perceptual_loss = self.perceptual_loss(imgs, decoded_images)
        rec_loss = torch.abs(imgs - decoded_images)
        perceptual_rec_loss = (
            self.perceptual_loss_factor * perceptual_loss
            + self.rec_loss_factor * rec_loss
        )

        perceptual_rec_loss = perceptual_rec_loss.mean()

        """
        =======================================================================================================================
        Discriminator Loss
        """
        # ----- GAN loss G ----- #
        #disc_real = self.discriminator(imgs)
        disc_fake_g = self.discriminator(decoded_images)
        g_loss = -torch.mean(disc_fake_g)

        # ----- adaptive weight ----- #
        disc_factor = self.vqgan.adopt_weight(
            self.disc_factor, self.global_step, threshold=self.disc_start
        )
        λ = self.vqgan.calculate_lambda(perceptual_rec_loss, g_loss)
        vq_loss = perceptual_rec_loss + q_loss + disc_factor * λ * g_loss

        # ----- GAN loss D ----- #
        disc_real = self.discriminator(imgs)
        disc_fake_d = self.discriminator(decoded_images.detach())
        d_loss_real = torch.mean(F.relu(1.0 - disc_real))
        d_loss_fake = torch.mean(F.relu(1.0 + disc_fake_d))
        gan_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)

        # ======================================================================================================================
        # Tracking metrics

        self.run.log(
            {"Perceptual & Reconstruction loss":perceptual_rec_loss.item(),
            },
            step=self.global_step,
        )

        self.run.log(
            {"VQ loss":vq_loss.item()}, step=self.global_step
        )
        self.run.log(
            {"GAN loss":gan_loss.item()}, step=self.global_step
        )

        # ----- scale loss by accumulation ----- #
        vq_loss = vq_loss / self.update_every
        gan_loss = gan_loss / self.update_every

        # =======================================================================================================================
        # backforward pass

        #breakpoint()
        #self.opt_vq.zero_grad()
        vq_loss.backward(
            retain_graph=True
        )  # retain_graph is used to retain the computation graph for the discriminator loss # FIXME: this might accumulate gradients and lead to OOM
        if (idx +1) % self.update_every == 0:
            self.opt_vq.step()
            self.opt_vq.zero_grad()

        #self.opt_disc.zero_grad()
        gan_loss.backward()
        if (idx + 1) % self.update_every == 0:
            self.opt_disc.step()
            self.opt_disc.zero_grad() 

        #self.opt_vq.step()
        #self.opt_disc.step()

        return decoded_images, vq_loss, gan_loss
    def backward(self, imgs: torch.Tensor) -> None:
        # =======================================================================================================================
        # backpropagation 
        self.opt_vq.step()
        self.opt_vq.zero_grad()
        self.opt_disc.step()
        self.opt_disc.zero_grad()

    def train(
        self,
        dataloader: torch.utils.data.DataLoader,
        epochs: int = 1,
    ):
        """Trains the VQGAN for the given number of epochs

        Args:
            dataloader (torch.utils.data.DataLoader): dataloader to use.
            epochs (int, optional): number of epochs to train for. Defaults to 100.
        """

        print(f"Training VQGAN with {count_parameters(self.vqgan)/(1024**2)} M parameters")
        for epoch in range(epochs):
            for index, imgs in enumerate(dataloader):

                if index % self.update_every == 0:
                    img_list, decoded_image_list = [], []
                # Training step
                imgs = imgs.to(self.device)
                img_list.append(imgs)
                

                decoded_images, vq_loss, gan_loss = self.step(index, imgs)
                #if (index + 1) % self.update_every == 0:
                #    self.backward()
                decoded_image_list.append(decoded_images)

                # Updating global step
                self.global_step += 1

                if (index + 1) % self.save_every == 0 and (index + 1) % self.update_every == 0:
                    decoded_image_list: torch.Tensor = torch.cat(decoded_image_list, dim=0)
                    img_list: torch.Tensor = torch.cat(img_list, dim=0)

                    print(
                        f"Epoch: {epoch+1}/{epochs} | Batch: {index}/{len(dataloader)} | VQ Loss : {vq_loss:.4f} | Discriminator Loss: {gan_loss:.4f}"
                    )

                    # Only saving the gif for the first 2000 save steps
                    if self.global_step // self.save_every <= 2000:
                        self.sample_batch = (
                            imgs[:] if self.sample_batch is None else self.sample_batch
                        )

                        with torch.no_grad():
                            
                            """
                            Note : Lots of efficiency & cleaning needed here
                            """

                            gif_img = (
                                torchvision.utils.make_grid(
                                    torch.cat(
                                        (
                                            self.sample_batch,
                                            self.vqgan(self.sample_batch)[0],
                                        ),
                                    )
                                )
                                .detach()
                                .cpu()
                                .permute(1, 2, 0)
                                .numpy()
                            )

                            # image norm before logging.
                            gif_img = (gif_img - gif_img.min()) * (
                                255 / (gif_img.max() - gif_img.min())
                            )
                            gif_img = gif_img.astype(np.uint8)

                            self.run.log(
                                {
                                    "VQGAN Recon":
                                    Image(
                                        torchvision.utils.make_grid(
                                            
                                            decoded_image_list,
                                            normalize=True,
                                        ).permute(1,2,0).cpu().numpy(),
                                        caption= "VQGAN Reconstrucrtion",
                                    ),
                                    "VQGAN GT":
                                    Image(
                                        torchvision.utils.make_grid(
                                            
                                            img_list,
                                            normalize=True,
                                        ).permute(1,2,0).cpu().numpy(),
                                        caption= "VQGAN GT",
                                    ),
                                },
                                    step=self.global_step,
                            )

                            self.gif_images.append(gif_img)

                        imageio.mimsave(
                            os.path.join(self.expriment_save_dir, "vqgan-reconstruction.gif"),
                            self.gif_images,
                            fps=5,
                        )
