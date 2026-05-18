"""GAN training script for EDNIG (PyTorch port).

Run on a CUDA-enabled machine:

    python -m pytorch_impl.train --data ./data/lol_dataset --img-size 512 --epochs 180

This mirrors the original paper recipe:
- Generator + Critic (WGAN-like)
- Each generator step is preceded by `critic_updates` critic steps
- Generator loss = 100 * (perceptual + 10 * L2) + 1 * (-critic_score)
- Optimizer: Adam with lr 1e-4, beta1=0.9, beta2=0.999, linear LR decay
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import LOLDataset
from .losses import PerceptualAndL2Loss, VGGPerceptual, wasserstein_loss
from .model import Discriminator, Generator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="./data/lol_dataset",
                   help="Root directory containing our485/ and eval15/")
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=180)
    p.add_argument("--critic-updates", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save-dir", type=str, default="./weights")
    p.add_argument("--save-every", type=int, default=5,
                   help="Save checkpoint every N epochs.")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--resume-g", type=str, default="",
                   help="Optional path to generator .pt to resume from.")
    p.add_argument("--resume-d", type=str, default="",
                   help="Optional path to discriminator .pt to resume from.")
    p.add_argument("--device", type=str, default="cuda",
                   help="cuda or cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        print("[WARN] CUDA requested but not available, falling back to CPU.")

    save_dir = Path(args.save_dir)
    (save_dir / "g").mkdir(parents=True, exist_ok=True)
    (save_dir / "d").mkdir(parents=True, exist_ok=True)

    # Data
    ds = LOLDataset(args.data, split="train", img_size=args.img_size, augment=True)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    print(f"[INFO] Training samples: {len(ds)} | iters/epoch: {len(loader)}")

    # Models
    G = Generator().to(device)
    D = Discriminator().to(device)
    if args.resume_g:
        G.load_state_dict(torch.load(args.resume_g, map_location=device))
        print(f"[INFO] Resumed G from {args.resume_g}")
    if args.resume_d:
        D.load_state_dict(torch.load(args.resume_d, map_location=device))
        print(f"[INFO] Resumed D from {args.resume_d}")

    # Optimizers (linear decay over training, similar to original)
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    decay_rate = (args.lr / args.epochs) * 0.5

    # Loss helpers
    vgg = VGGPerceptual().to(device).eval()
    p_l2_loss = PerceptualAndL2Loss(a=1.0, b=10.0, vgg=vgg).to(device)

    # Critic ground-truth labels
    label_real = torch.ones(args.batch_size, 1, device=device)
    label_fake = -torch.ones(args.batch_size, 1, device=device)

    g_weight, c_weight = 100.0, 1.0

    for epoch in range(1, args.epochs + 1):
        # Linear LR decay
        lr_now = max(args.lr - decay_rate * (epoch - 1), 1e-6)
        for pg in opt_d.param_groups:
            pg["lr"] = lr_now
        for pg in opt_g.param_groups:
            pg["lr"] = lr_now

        t0 = time.time()
        d_losses, g_losses = [], []
        G.train(); D.train()
        for step, (inp, tgt) in enumerate(loader):
            inp = inp.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)

            with torch.no_grad():
                fake = G(inp)

            # ---- Critic updates ----
            for _ in range(args.critic_updates):
                opt_d.zero_grad(set_to_none=True)
                d_real = D(tgt)
                d_fake = D(fake.detach())
                loss_d = 0.5 * (
                    wasserstein_loss(d_real, label_real)
                    + wasserstein_loss(d_fake, label_fake)
                )
                loss_d.backward()
                opt_d.step()
                d_losses.append(loss_d.item())

            # ---- Generator update ----
            opt_g.zero_grad(set_to_none=True)
            fake = G(inp)
            d_on_fake = D(fake)
            loss_g_content = p_l2_loss(fake, tgt)
            # The critic outputs "real-ness" — generator wants D(fake) -> +1
            # Adversarial term equivalent to wasserstein_loss(D(fake), +1):
            loss_g_adv = wasserstein_loss(d_on_fake, label_real)
            loss_g = g_weight * loss_g_content + c_weight * loss_g_adv
            loss_g.backward()
            opt_g.step()
            g_losses.append(loss_g.item())

            if (step + 1) % 25 == 0:
                print(
                    f"  [ep {epoch}/{args.epochs}] step {step+1}/{len(loader)} | "
                    f"d_loss={sum(d_losses[-args.critic_updates:])/max(1,len(d_losses[-args.critic_updates:])):.4f} "
                    f"g_loss={loss_g.item():.4f} (content={loss_g_content.item():.4f} adv={loss_g_adv.item():.4f})"
                )

        dt = time.time() - t0
        mean_d = sum(d_losses) / max(1, len(d_losses))
        mean_g = sum(g_losses) / max(1, len(g_losses))
        print(
            f"[Epoch {epoch}/{args.epochs}] lr={lr_now:.6f} "
            f"D_loss={mean_d:.4f} G_loss={mean_g:.4f} time={dt:.1f}s"
        )

        # Save checkpoints
        if epoch % args.save_every == 0 or epoch == args.epochs:
            g_path = save_dir / "g" / f"generator_ep{epoch:03d}.pt"
            d_path = save_dir / "d" / f"discriminator_ep{epoch:03d}.pt"
            torch.save(G.state_dict(), g_path)
            torch.save(D.state_dict(), d_path)
            # Always overwrite a 'latest' convenience file
            torch.save(G.state_dict(), save_dir / "ednig_generator_latest.pt")
            print(f"  [SAVED] {g_path.name}, {d_path.name}")

    print("[DONE] Training complete.")


if __name__ == "__main__":
    main()
